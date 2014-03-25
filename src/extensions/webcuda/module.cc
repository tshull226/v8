#include "module.h"
#include "function.h"

using namespace v8;
using namespace webcuda;

Persistent<ObjectTemplate> Module::constructor_template;

void Module::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "Module"),
			FunctionTemplate::New(isolate, MakeModuleObject));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "moduleLoad"),
			FunctionTemplate::New(isolate, Load));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "getFunction"),
			FunctionTemplate::New(isolate, GetFunction));

	Handle<ObjectTemplate> raw_template = MakeModuleTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

  //target->Set(String::NewSymbol("Module"), constructor_template->GetFunction());
}


Handle<ObjectTemplate> Module::MakeModuleTemplate(Isolate* isolate) {
  EscapableHandleScope handle_scope(isolate);

  Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
  result->SetInternalFieldCount(1);

  // Again, return the result through the current handle scope.
  return handle_scope.Escape(result);
}

Handle<Object> Module::MakeModuleObject_(Isolate* isolate) {
	EscapableHandleScope handle_scope(isolate);
	
	Module* pmodule = new Module();
	//cuDeviceGet(&(pdevice->m_device), deviceNum);

	//creating object
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(isolate, constructor_template);
	Local<Object> result = templ->NewInstance();

	//wrap the raw C++ pointer in an External so it can be referenced from within JavaScript
	Handle<External> module_ptr = External::New(isolate, pmodule);

	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, module_ptr);

	return handle_scope.Escape(result);
}

Module* Module::UnwrapModule(Handle<Object> obj) {
	Handle<External> field = Handle<External>::Cast(obj->GetInternalField(0));
	void* ptr = field->Value();
	return static_cast<Module*>(ptr);
}

void  Module::MakeModuleObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	Handle<Object> result = MakeModuleObject_(args.GetIsolate());

	args.GetReturnValue().Set(result);
}

void Module::Load(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	/*
	Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
	Module *pmodule = ObjectWrap::Unwrap<Module>(result);
	*/
	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	//I THINK THIS NEEDS TO BE FIXED
	String::Utf8Value fname(args[0]);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

void Module::GetFunction(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	Local<Object> result = webcuda::Function::MakeFunctionObject(args.GetIsolate());
	webcuda::Function *pfunction = webcuda::Function::UnwrapFunction(result);
	Module *pmodule = UnwrapModule(Handle<Object>::Cast(args[0]));
	//NodeCuda::Function *pfunction = ObjectWrap::Unwrap<NodeCuda::Function>(result);

	String::Utf8Value name(args[1]);
	CUresult error = cuModuleGetFunction(&(pfunction->m_function), pmodule->m_module, *name);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "name"), args[1]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

