#ifndef FUNCTION_H
#define FUNCTION_H

#include <cuda.h>
#include "bindings.h"
#include "webcuda.h"
#include "object_wrap.h"

using namespace v8;
using v8::internal::Arguments;

namespace WebCuda {
	class WebCUDA;

  class Function : public ObjectWrap {
    public:
      static void Initialize(Handle<Object> target);

    protected:
      static Persistent<FunctionTemplate> constructor_template;

      static Handle<Value> LaunchKernel(const Arguments& args);

      Function() : ObjectWrap(), m_function(0) {}

      ~Function() {}

    private:
      static Handle<Value> New(const Arguments& args);

      CUfunction m_function;

      friend Handle<Value> WebCUDA::GetFunction(const Arguments&);
  };

}

#endif
