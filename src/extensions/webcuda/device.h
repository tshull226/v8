/** \file device.h */

#ifndef DEVICE_H
#define DEVICE_H

#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda {

/**
 * \class Device
 * \brief information about CUDA Device
 *
 * add more stuff
 */
  class Device {
    public:
			/**
			 * \brief add Device classes features into JavaScript Object environment
			 */
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

			static Device* UnwrapDevice(Handle<Object> obj);
      CUdevice m_device;
    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static void  MakeDeviceObject(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void GetComputeCapability(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);
      static void GetName(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);
      static void GetTotalMem(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);


      // TODO: cuDeviceGetAttribute
      // TODO: cuDeviceGetProperties

      Device() : m_device(0) {}

      ~Device() {}

    private:
      static Handle<ObjectTemplate> MakeDeviceTemplate(Isolate* isolate);

  };

}

#endif
