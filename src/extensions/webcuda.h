#ifndef WEBCUDA_H
#define WEBCUDA_H

#include <cuda.h>
#include "object_wrap.h"
#include "bindings.h"
#include "webcuda_top.h"

using namespace v8;
using v8::internal::Arguments;

namespace WebCuda {

  class WebCUDA : public ObjectWrap {
    public:
      static void Initialize(Handle<Object> target);
      static Handle<Value> GetFunction(const Arguments& args);

    protected:
      static Persistent<FunctionTemplate> constructor_template;

      static Handle<Value> Load(const Arguments& args);

      WebCUDA() : ObjectWrap(), m_module(0) {}

      ~WebCUDA() {}

    private:
      static Handle<Value> New(const Arguments& args);

      CUmodule m_module;
  };

}

#endif
