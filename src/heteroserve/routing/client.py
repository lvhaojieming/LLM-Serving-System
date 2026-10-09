"""One Gateway dispatches probability RPCs across Ready headless-Service replicas."""
import asyncio
import hashlib
import ipaddress
import math
import socket
import time
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from .runtime import RoutingDecision


class RemoteRouterClient:
    remote = True

    def __init__(self, settings, config, client, resolver=None):
        self.settings, self.config, self.client = settings, config, client
        self.resolver = resolver or self.resolve
        self.addresses, self.active = [], {}
        self.updated = None
        self.cursor = 0
        self.last_identity = None

    async def resolve(self):
        parsed = urlparse(self.config["base_url"])
        if parsed.scheme != "http" or not parsed.hostname or parsed.path not in {"", "/"}:
            raise ValueError("L1 discovery requires an internal HTTP headless Service")
        port = parsed.port or 80
        rows = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, port, 0, socket.SOCK_STREAM)
        return sorted({f"http://[{ipaddress.ip_address(row[4][0])}]:{port}" if ':' in row[4][0] else f"http://{ipaddress.ip_address(row[4][0])}:{port}" for row in rows})

    async def endpoints(self):
        now = time.monotonic()
        if self.updated is None or now - self.updated >= self.config.get("refresh_seconds", 2):
            try:
                addresses = await self.resolver()
                if not addresses:
                    raise OSError("Headless Service has no Ready endpoints")
                self.addresses, self.updated = addresses, now
            except (OSError, ValueError):
                if self.updated is None or now - self.updated > self.config.get("cache_seconds", 15):
                    raise HTTPException(503, "L1 discovery is unavailable or expired")
        return list(self.addresses)

    def validate_identity(self, value):
        if value.get("checkpoint_sha256") != self.settings.checkpoint_sha256 or value.get("chat_template_sha256") != self.config["chat_template_sha256"]:
            raise HTTPException(503, "L1 checkpoint/template contract mismatch")
        if not all(isinstance(value.get(k), str) and value[k] for k in ("pod_uid", "node", "instance_id")):
            raise HTTPException(503, "L1 response lacks instance identity")
        if not all(isinstance(value.get(k), str) and value[k].startswith("npu:") for k in ("encoder_device", "head_device")):
            raise HTTPException(503, "L1 encoder/classifier device contract mismatch")

    async def ready(self):
        try:
            for address in await self.endpoints():
                try:
                    response = await self.client.get(address + "/ready", timeout=3)
                    if response.status_code == 200:
                        self.validate_identity(response.json())
                        return True
                except (httpx.HTTPError, ValueError):
                    continue
        except HTTPException:
            pass
        return False

    async def route(self, messages, generation, kwargs):
        endpoints = await self.endpoints()
        offset = self.cursor % len(endpoints)
        self.cursor += 1
        rotated = endpoints[offset:] + endpoints[:offset]
        candidates = sorted(rotated, key=lambda address: self.active.get(address, 0))
        overloaded = False
        for address in candidates:
            self.active[address] = self.active.get(address, 0) + 1
            try:
                response = await self.client.post(address + "/route", json={"messages": messages, "max_new_tokens": generation, "template_kwargs": kwargs}, timeout=self.config.get("timeout_seconds", 30))
                if response.status_code in {429, 503}:
                    overloaded |= response.status_code == 429
                    continue
                if response.status_code == 400:
                    raise ValueError(response.json().get("detail", "L1 rejected the prompt"))
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as error:
                    raise HTTPException(503, "Malformed L1 JSON response") from error
                self.validate_identity(data["identity"])
                template = data["chat_template"]
                if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != self.config["chat_template_sha256"]:
                    raise HTTPException(503, "L1 returned a different canonical template")
                decision = RoutingDecision(**data["decision"])
                if decision.expert not in self.settings.expert_mapping.values() or set(decision.probabilities) != set(self.settings.expert_mapping.values()) or not all(math.isfinite(v) and 0 <= v <= 1 for v in decision.probabilities.values()) or not math.isclose(sum(decision.probabilities.values()), 1, abs_tol=1e-4) or type(decision.input_tokens) is not int or decision.input_tokens < 1 or not math.isfinite(decision.elapsed_ms) or decision.elapsed_ms < 0:
                    raise HTTPException(503, "L1 returned an invalid probability decision")
                self.chat_template, self.last_identity = template, data["identity"]
                return decision
            except httpx.HTTPError:
                continue
            except (KeyError, TypeError) as error:
                raise HTTPException(503, "Malformed L1 protocol response") from error
            finally:
                self.active[address] -= 1
        raise HTTPException(429 if overloaded else 503, "No L1 instance accepted the routing request", headers={"Retry-After": "1"})
