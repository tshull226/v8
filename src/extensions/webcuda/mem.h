#ifndef MEM_HPP
#define MEM_HPP

#include <cuda.h>
#include <v8.h>
//#include "function.hpp" not sure why it would need this...
using namespace v8;

namespace webcuda {

  class Mem {
    public:
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static Handle<Object> MakeMemObject(Isolate* isolate);
      static Handle<ObjectTemplate> MakeMemTemplate(Isolate* isolate);

      static void Alloc(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void AllocPitch(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void Free(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void CopyHtoD(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void CopyDtoH(const v8::FunctionCallbackInfo<v8::Value>& args);

			static Mem* UnwrapDevicePtr(Handle<Object> obj);

      //static Handle<Value> GetDevicePtr(Local<String> property, const AccessorInfo &info);

      Mem() : m_devicePtr(0) {}

      ~Mem() {}

    private:

      CUdeviceptr m_devicePtr;

//      friend class NodeCuda::Function; NEED TO SEE WHY THIS IS HERE...
  };

}

#endif
