#include "bindings.h"
#include "ctx.h"
#include "device.h"
#include "function.h"
#include "mem.h"
#include "webcuda.h"

using namespace WebCuda;
//using namespace v8;
using namespace v8::internal;

void init (Handle<Object> target) {
  HandleScope scope;

  // Initiailze the cuda driver api
  cuInit(0);

  // These methods don't need instances
  target->SetAccessor(String::New("driverVersion"), GetDriverVersion);
  target->SetAccessor(String::New("deviceCount"), GetDeviceCount);

  // Initialize driver api bindings
  Ctx::Initialize(target);
  Device::Initialize(target);
  WebCuda::Function::Initialize(target);
  Mem::Initialize(target);
  WebCUDA::Initialize(target);
}

Handle<Value> WebCuda::GetDriverVersion(Local<String> property, const AccessorInfo &info) {
  HandleScope scope;
  int driverVersion = 0;
  cuDriverGetVersion(&driverVersion);
  return scope.Close(Integer::New(driverVersion));
}

Handle<Value> WebCuda::GetDeviceCount(Local<String> property, const AccessorInfo &info) {
  HandleScope scope;
  int count = 0;
  cuDeviceGetCount(&count);
  return scope.Close(Integer::New(count));
}

/*going to have to have to do something else to initialize*/
//NODE_MODULE(cuda, init);
