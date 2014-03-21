#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <cuda.h>
#include "bindings.h"


using namespace v8;

namespace WebCuda {

  class Device : public ObjectWrap {
    public:
      static void Initialize(Handle<Object> target);

    protected:
      static Persistent<FunctionTemplate> constructor_template;

      static Handle<Value> New(const internal::Arguments& args);
      static Handle<Value> GetComputeCapability(Local<String> property, const internal::AccessorInfo &info);
      static Handle<Value> GetName(Local<String> property, const internal::AccessorInfo &info);
      static Handle<Value> GetTotalMem(Local<String> property, const internal::AccessorInfo &info);

      // TODO: cuDeviceGetAttribute
      // TODO: cuDeviceGetProperties

      //Device() : ObjectWrap(), m_device(NULL) {}
      Device() : ObjectWrap(), m_device(0) {}

      ~Device() {}

    private:
      CUdevice m_device;

      friend class Ctx;
  };

}

#endif
