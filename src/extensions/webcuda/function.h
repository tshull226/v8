#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cuda.h>
#include <v8.h>
//#include "module.hpp" need to see why i need this

using namespace v8;

namespace webcuda {

  class Function {
    public:
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static void  LaunchKernel(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void  MakeFunctionObject(const v8::FunctionCallbackInfo<v8::Value>& args);

      Function() : m_function(0) {}

      ~Function() {}

    private:
      //static Handle<Value> New(const Arguments& args); shouldn't need this
      static Handle<Object> MakeFunctionObject_(Isolate* isolate);
      static Handle<ObjectTemplate> MakeFunctionTemplate(Isolate* isolate);
      static Function* UnwrapFunction(Handle<Object> obj);

      CUfunction m_function;

      //friend Handle<Value> Module::GetFunction(const Arguments&); see why i need this...
  };

}

#endif
