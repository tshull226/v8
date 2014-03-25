#ifndef MODULE_HPP
#define MODULE_HPP

#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda {

  class Module {
    public:
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);
      static void  GetFunction(const v8::FunctionCallbackInfo<v8::Value>& args);

    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static void  Load(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void  MakeModuleObject(const v8::FunctionCallbackInfo<v8::Value>& args);

			static Module* UnwrapModule(Handle<Object> obj);

      Module() : m_module(0) {}

      ~Module() {}

    private:
 //     static Handle<Value> New(const Arguments& args);
      static Handle<Object>  MakeModuleObject_(Isolate* isolate);
      static Handle<ObjectTemplate> MakeModuleTemplate(Isolate* isolate);

      CUmodule m_module;
  };

}

#endif
