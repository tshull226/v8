#include "webcuda.h"
#include "function.h"

using namespace WebCuda;
using namespace v8;

Persistent<FunctionTemplate> WebCUDA::constructor_template;

void WebCUDA::Initialize(Handle<Object> target) {
  HandleScope scope;

  Local<FunctionTemplate> t = FunctionTemplate::New(WebCUDA::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("WebCUDAModule"));

  // WebCUDA objects can only be created by load functions
  WEBCUDA_SET_METHOD(target, "moduleLoad", WebCUDA::Load);

  WEBCUDA_SET_PROTOTYPE_METHOD(constructor_template, "getFunction", WebCUDA::GetFunction);
}

Handle<Value> WebCUDA::New(const Arguments& args) {
  HandleScope scope;

  WebCUDA *pmem = new WebCUDA();
  pmem->Wrap(args.This());

  return args.This();
}

Handle<Value> WebCUDA::Load(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
  WebCUDA *pmodule = ObjectWrap::Unwrap<WebCUDA>(result);

  String::AsciiValue fname(args[0]);
  CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

  result->Set(String::New("fname"), args[0]);
  result->Set(String::New("error"), Integer::New(error));

  return scope.Close(result);
}

Handle<Value> WebCUDA::GetFunction(const Arguments& args) {
  HandleScope scope;
  Local<Object> result = WebCUDA::Function::constructor_template->InstanceTemplate()->NewInstance();
  WebCUDA *pmodule = ObjectWrap::Unwrap<WebCUDA>(args.This());
  WebCUDA::Function *pfunction = ObjectWrap::Unwrap<WebCUDA::Function>(result);

  String::AsciiValue name(args[0]);
  CUresult error = cuModuleGetFunction(&(pfunction->m_function), pmodule->m_module, *name);

  result->Set(String::New("name"), args[0]);
  result->Set(String::New("error"), Integer::New(error));

  return scope.Close(result);
}

