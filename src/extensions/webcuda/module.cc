#include "module.h"
#include "function.h"
#include <stdlib.h>
#include <iostream>

using namespace v8;
using namespace webcuda;
using std::cout;
using std::endl;
using std::string;

Persistent<ObjectTemplate> Module::constructor_template;

void Module::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "Module"),
			FunctionTemplate::New(isolate, MakeModuleObject));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "moduleLoad"),
			FunctionTemplate::New(isolate, Load));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "getFunction"),
			FunctionTemplate::New(isolate, GetFunction));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "compile"),
			FunctionTemplate::New(isolate, Compile));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "compileFile"),
			FunctionTemplate::New(isolate, CompileFile));

	Handle<ObjectTemplate> raw_template = MakeModuleTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

  //target->Set(String::NewSymbol("Module"), constructor_template->GetFunction());
}


/**
 *
 */
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

/* \param args the name of the kernel function to load
 *
 * uses cuModleLoad to load a kernel into a JavaScript object from which
 * functions can be extracted from
 */
void Module::Load(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	/*
	Local<Object> result = constructor_template->InstanceTemplate()->NewInstance();
	Module *pmodule = ObjectWrap::Unwrap<Module>(result);
	*/
	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	String::Utf8Value fname(args[0]);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

#define NVCC "/Developer/NVIDIA/CUDA-5.5/bin/nvcc"
#define NVCC_FLAGS ""

/** \param kFile name of text file containing CUDA kernel
 * 
 * Invokes NVCC on a text file by using system calls to compile a .cu file into
 * a .ptx file. Returns the name of the .ptx file created
 */
std::string Module::InvokeNVCC_(std::string kFile){
	std::string cuFile = "~/testing/test.ptx";
	kFile = "~/testing/test.cu";
	cout <<"trying" << endl;
	int nvcc_exit_status = std::system((std::string(NVCC) + " -ptx " + NVCC_FLAGS + " " + kFile + " -o " + cuFile).c_str());

	if (nvcc_exit_status != 0){
		cout << "no go" << endl;
	}

	return cuFile;
}

/** \param args string containing CUDA code 
 * 
 * WebCUDA Wrapper (webcuda.compileFile) to Compile a CUDA string and return a
 * WebCUDA Module object. Uses InvokeNVCC_ to call nvcc and creates a cubin from
 * which cuModuleLoad can be used to extract the module.
 */
void Module::Compile(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	String::Utf8Value fname(args[0]);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

/** \param args name of the filename containing the CUDA code
 * 
 * WebCUDA Wrapper (webcuda.compile) to Compile file and return a WebCUDA Module
 * object. Uses InvokeNVCC_ to call nvcc and creates a cubin from which
 * cuModuleLoad can be used to extract the module.
 */
void Module::CompileFile(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	InvokeNVCC_("");

	//args.GetReturnValue().Set(result);
}

/** \param args name of the kernel function to extract from module
 *
 * WebCUDA Wrapper for cuModuleGetFunction that retrieves a function from the
 * module and wraps it in a JavaScript Object. This object, in addition to the
 * wrapped CUmodule, contains fields denoting the name of the module and if an
 * error occured in creation. This is an property of WebCUDA Module object
 * wrappers
 */
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

