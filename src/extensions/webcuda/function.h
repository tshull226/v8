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
      static Handle<Object> MakeFunctionObject(Isolate* isolate) { return MakeFunctionObject_(isolate); }
      static Function* UnwrapFunction(Handle<Object> obj);
      CUfunction m_function;


    protected:
      static Persistent<ObjectTemplate> constructor_template;

      static void  MakeFunctionObject(const v8::FunctionCallbackInfo<v8::Value>& args);
      static void  LaunchKernel(const v8::FunctionCallbackInfo<v8::Value>& args);

      Function() : m_function(0) {}

      ~Function() {}

    private:
      //static Handle<Value> New(const Arguments& args); shouldn't need this
      static Handle<Object> MakeFunctionObject_(Isolate* isolate);
      static Handle<ObjectTemplate> MakeFunctionTemplate(Isolate* isolate);


      //friend Handle<Value> Module::GetFunction(const Arguments&); see why i need this...
  };

}

#endif
