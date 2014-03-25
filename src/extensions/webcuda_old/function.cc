//TOM need to adjust this method
//#include <node_buffer.h>
#include <cstring>
#include <cstdio>
#include "function.h"
#include "mem.h"

using namespace WebCuda;

Persistent<FunctionTemplate> WebCuda::Function::constructor_template;

void WebCuda::Function::Initialize(Handle<Object> target) {
  HandleScope scope;

  Local<FunctionTemplate> t = FunctionTemplate::New(WebCuda::Function::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("CudaFunction"));

  WEBCUDA_SET_PROTOTYPE_METHOD(constructor_template, "launchKernel", WebCuda::Function::LaunchKernel);

  // Function objects can only be created by cuModuleGetFunction
}

Handle<Value> WebCuda::Function::New(const Arguments& args) {
  HandleScope scope;

  WebCuda::Function *pfunction = new WebCuda::Function();
  pfunction->Wrap(args.This());

  return args.This();
}

Handle<Value> WebCuda::Function::LaunchKernel(const Arguments& args) {
  HandleScope scope;
  Function *pfunction = ObjectWrap::Unwrap<Function>(args.This());

  Local<Array> gridDim = Local<Array>::Cast(args[0]);
  unsigned int gridDimX = gridDim->Get(0)->Uint32Value();
  unsigned int gridDimY = gridDim->Get(1)->Uint32Value();
  unsigned int gridDimZ = gridDim->Get(2)->Uint32Value();

  Local<Array> blockDim = Local<Array>::Cast(args[1]);
  unsigned int blockDimX = blockDim->Get(0)->Uint32Value();
  unsigned int blockDimY = blockDim->Get(1)->Uint32Value();
  unsigned int blockDimZ = blockDim->Get(2)->Uint32Value();

  Local<Object> buf = args[2]->ToObject();
	//TOM_FIX
	/*
	 * in node_buffer.h lines 40-43
  char *pbuffer = Buffer::Data(buf);
  size_t bufferSize = Buffer::Length(buf);
	*/
	//TOM TEMP SOLUTION
  char *pbuffer = 0;
  size_t bufferSize = 0;

  void *cuExtra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, pbuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &bufferSize,
    CU_LAUNCH_PARAM_END
  };

  CUresult error = cuLaunchKernel(pfunction->m_function,
      gridDimX, gridDimY, gridDimZ,
      blockDimX, blockDimY, blockDimZ,
      0, 0, NULL, cuExtra);

  return scope.Close(Number::New(error));
}

