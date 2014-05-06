#include "module.h"
#include "function.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

using namespace v8;
using namespace webcuda;
using std::cout;
using std::endl;
using std::string;
using std::ofstream;
using std::stringstream;

Persistent<ObjectTemplate> Module::constructor_template;

void Module::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "moduleLoad"),
			FunctionTemplate::New(isolate, Load));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "moduleUnload"),
			FunctionTemplate::New(isolate, Unload));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "getFunction"),
			FunctionTemplate::New(isolate, GetFunction));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "compile"),
			FunctionTemplate::New(isolate, Compile));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "compileFile"),
			FunctionTemplate::New(isolate, CompileFile));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "compileText"),
			FunctionTemplate::New(isolate, CompileText));

	Handle<ObjectTemplate> raw_template = MakeModuleTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

	//initializing the random number generator
	//std::srand(std::time(NULL));
	//ACUTALLY LIKE THE DETERMINSTIC ASPECT OF IT FOR THE TIME BEING

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

	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	String::Utf8Value fname(args[0]);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

/* \param args the name of the kernel function to unload
 *
 * uses cuModuleUnload to unload a kernel into a JavaScript object from which
 * functions can be extracted from. Also frees any temporarily created files used for compilation
 */
void Module::Unload(const v8::FunctionCallbackInfo<v8::Value>& args) {
	Handle<Object> module = Handle<Object>::Cast(args[0]);
	Handle<String> textHandle = String::NewFromUtf8(args.GetIsolate(), "textName");
	Handle<String> strHandle = String::NewFromUtf8(args.GetIsolate(), "cuString");
	Handle<String> cuHandle = String::NewFromUtf8(args.GetIsolate(), "cuName");
	Handle<String> fileHandle = String::NewFromUtf8(args.GetIsolate(), "fname");


	if(module->HasOwnProperty(textHandle) || module->HasOwnProperty(strHandle)){
		Handle<Value> temp = module->Get(cuHandle);
		String::Utf8Value fileValue1(temp);
		string deleteFile(*fileValue1);
		deleteFile += ".cu";
		std::system((std::string("rm ") + deleteFile).c_str());

		temp = module->Get(fileHandle);
		String::Utf8Value fileValue2(temp);
		deleteFile = *fileValue2;
		std::system((std::string("rm ") + deleteFile).c_str());
	} else if(module->HasOwnProperty(cuHandle)){
		Handle<Value> temp = module->Get(fileHandle);
		String::Utf8Value fileValue1(temp);
		string deleteFile(*fileValue1);
		std::system((std::string("rm ") + deleteFile).c_str());

	}

	Module *pmodule = UnwrapModule(module);
	CUresult error = cuModuleUnload((pmodule->m_module));

	args.GetReturnValue().Set(Integer::New(args.GetIsolate(), error));

}

#ifdef __APPLE__
#define NVCC "/Developer/NVIDIA/CUDA-5.5/bin/nvcc"
#else
#define NVCC "/usr/local/cuda-5.5/bin/nvcc"
#endif

#define NVCC_FLAGS ""

/** \param kFile name of text file containing CUDA kernel
 * 
 * Invokes NVCC on a text file by using system calls to compile a .cu file into
 * a .ptx file. Returns the name of the .ptx file created
 */
Handle<Object>  Module::InvokeNVCC_(Isolate *isolate, Handle<Value> kFileHandle){

	EscapableHandleScope scope(isolate);

	String::Utf8Value kFile(kFileHandle);
	std::string cuFile(*kFile);
	cuFile +=	".cu";
	std::string outFile(*kFile);
	outFile += ".ptx";
#ifdef V8_WEBCUDA_DEBUG
	cout <<"trying" << endl;
#endif
	int nvcc_exit_status = std::system((std::string(NVCC) + " -ptx " + NVCC_FLAGS + " " + cuFile + " -o " + outFile).c_str());

	if (nvcc_exit_status != 0){
#ifdef V8_WEBCUDA_DEBUG
		cout << "no go" << endl;
#endif
	}

	Local<Object> returnValue = Local<Object>::Cast(String::NewFromUtf8(isolate, outFile.c_str()));
	return scope.Escape(returnValue);
}

/** \param args string containing CUDA code 
 * 
 * WebCUDA Wrapper (webcuda.compileFile) to Compile a CUDA string and return a
 * WebCUDA Module object. Uses InvokeNVCC_ to call nvcc and creates a cubin from
 * which cuModuleLoad can be used to extract the module.
 */
void Module::Compile(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	//creating module
	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	//creating temp file for compilation
	//random file name
	int random_val = std::rand() % 10000;
	stringstream ss;
	ss << random_val;
	random_val = std::rand() % 10000;
	ss << "__" << random_val;
	string cuFileName = ss.str();

	//write CUDA text to random file name
	//retrieving CUDA string
	String::Utf8Value cuString(args[0]);
	ofstream cuFile;
	cuFile.open((cuFileName + ".cu").c_str());
	cuFile << *cuString;
	cuFile.close();

	//compiling
	Handle<Value> cuNameHandle = Handle<Value>::Cast(String::NewFromUtf8(args.GetIsolate(), cuFileName.c_str()));
	//creating temp file for compilation and compiling
	Handle<Object> fnameHandle = InvokeNVCC_(args.GetIsolate(), cuNameHandle);
	String::Utf8Value fname(fnameHandle);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "cudaString"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "cuName"), cuNameHandle);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), fnameHandle);
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

	//creating module
	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	//creating temp file for compilation and compiling
	Handle<Object> fnameHandle = InvokeNVCC_(args.GetIsolate(), args[0]);
	String::Utf8Value fname(fnameHandle);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "cuName"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), fnameHandle);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

/** \param args name of the filename containing the CUDA code
 * 
 * WebCUDA Wrapper (webcuda.compile) to Compile file and return a WebCUDA Module
 * object. Uses InvokeNVCC_ to call nvcc and creates a cubin from which
 * cuModuleLoad can be used to extract the module.
 */
void Module::CompileText(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	//creating module
	Handle<Object> result = MakeModuleObject_(args.GetIsolate());
	Module *pmodule = UnwrapModule(result);

	//creating .cu file
	String::Utf8Value kFile(args[0]);
	string txtFile(*kFile);
	txtFile += ".txt";
	//creating cu file name
	int random_val = std::rand() % 10000;
	stringstream ss;
	ss << random_val;
	random_val = std::rand() % 10000;
	ss << "__" << random_val;
	string cuFile = ss.str();

	//copying text to cu file
	std::system((std::string("cp ") + txtFile + " " + cuFile + ".cu").c_str());
	Handle<Value> cuNameHandle = Handle<Value>::Cast(String::NewFromUtf8(args.GetIsolate(), cuFile.c_str()));
	//creating temp file for compilation and compiling
	Handle<Object> fnameHandle = InvokeNVCC_(args.GetIsolate(), cuNameHandle);
	String::Utf8Value fname(fnameHandle);
	CUresult error = cuModuleLoad(&(pmodule->m_module), *fname);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "textName"), args[0]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "cuName"), cuNameHandle);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "fname"), fnameHandle);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));


	args.GetReturnValue().Set(result);
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

#ifdef V8_WEBCUDA_DEBUG
	if(error == CUDA_ERROR_NOT_FOUND){
		cout << "could not find function " << *name << endl;
	} else if (error == CUDA_SUCCESS){
		cout << "successful?" << endl;
	} else{
		cout << "back to drawing board" << endl;
	}
#endif


	result->Set(String::NewFromUtf8(args.GetIsolate(), "name"), args[1]);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

