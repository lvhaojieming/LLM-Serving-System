// Stable C ABI over FlagCX; deliberately independent of either Torch ABI.
#include <flagcx.h>
#include <new>

struct Transport {
  flagcxDeviceHandle_t device = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxStream_t stream = nullptr;
  int index = 0;
};

extern "C" {
int hs_flagcx_open(void **out, int device) {
  auto *t = new (std::nothrow) Transport;
  if (!t) return flagcxSystemError;
  t->index = device;
  auto result = flagcxDeviceHandleInit(&t->device);
  if (result == flagcxSuccess) result = t->device->setDevice(device);
  if (result != flagcxSuccess) {
    if (t->device) flagcxDeviceHandleFree(t->device);
    delete t;
    return result;
  }
  *out = t;
  return flagcxSuccess;
}

int hs_flagcx_unique_id(void *id) {
  return flagcxGetUniqueId(static_cast<flagcxUniqueId_t>(id));
}

int hs_flagcx_init(void *context, void *id, int rank) {
  auto *t = static_cast<Transport *>(context);
  auto result = flagcxCommInitRank(&t->comm, 2,
                                  static_cast<flagcxUniqueId_t>(id), rank);
  if (result == flagcxSuccess) result = t->device->streamCreate(&t->stream);
  return result;
}

int hs_flagcx_is_homo(void *context, int *homo) {
  return flagcxIsHomoComm(static_cast<Transport *>(context)->comm, homo);
}

int hs_flagcx_transfer(void *context, void *buffer, size_t count,
                      int peer, int receive) {
  auto *t = static_cast<Transport *>(context);
  auto result = t->device->setDevice(t->index);
  if (result != flagcxSuccess) return result;
  result = receive ? flagcxRecv(buffer, count, flagcxBfloat16, peer,
                                t->comm, t->stream)
                   : flagcxSend(buffer, count, flagcxBfloat16, peer,
                                t->comm, t->stream);
  if (result == flagcxSuccess)
    result = t->device->streamSynchronize(t->stream);
  return result;
}

int hs_flagcx_close(void *context) {
  auto *t = static_cast<Transport *>(context);
  t->device->setDevice(t->index);
  auto result = flagcxSuccess;
  if (t->comm) result = flagcxCommDestroy(t->comm);
  if (t->stream) t->device->streamDestroy(t->stream);
  flagcxDeviceHandleFree(t->device);
  delete t;
  return result;
}

const char *hs_flagcx_error(int result) {
  return flagcxGetErrorString(static_cast<flagcxResult_t>(result));
}
}
