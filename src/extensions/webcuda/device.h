#ifndef DEVICE_H
#define DEVICE_H

#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda {

  class Device {
    public:
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static void  MakeDeviceObject(const v8::FunctionCallbackInfo<v8::Value>& args);
      static Handle<ObjectTemplate> MakeDeviceTemplate(Isolate* isolate);
      static void GetComputeCapability(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);
      static void GetName(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);
      static void GetTotalMem(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);

			static Device* UnwrapDevice(Handle<Object> obj);

      // TODO: cuDeviceGetAttribute
      // TODO: cuDeviceGetProperties

      Device() : m_device(0) {}

      ~Device() {}

    private:
      CUdevice m_device;

//      friend class Ctx; DON'T THINK THAT I NEED THIS ANYMORE
  };

}

#endif
