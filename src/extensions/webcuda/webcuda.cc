
#include <iostream>
#include <cuda.h>
#include <cudaProfiler.h>
#include "webcuda.h"
#include "device.h"

using namespace webcuda;
using namespace v8;
using std::cout;
using std::endl;

/**
 * calls initializers for other features (memory, device info, CUDA context creation, and kernal retrieval/launching)
 */
void WebCUDA::AddWebCUDAMethods(Isolate* isolate, Handle<ObjectTemplate> webcuda_templ){
	//instantiating basic webCUDA information
	webcuda_templ->Set(String::NewFromUtf8(isolate, "version"),
			FunctionTemplate::New(isolate, Version));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "driverVersion"),
			FunctionTemplate::New(isolate, GetDriverVersion));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "deviceCount"),
			FunctionTemplate::New(isolate, GetDeviceCount));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "startProfiling"),
			FunctionTemplate::New(isolate, webcuda::WebCUDA::StartProfiling));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "stopProfiling"),
			FunctionTemplate::New(isolate, webcuda::WebCUDA::StopProfiling));

	//instantiating other features
	Device::Initialize(isolate, webcuda_templ);
}

void WebCUDA::Version(const v8::FunctionCallbackInfo<v8::Value>& args){
#ifdef V8_WEBCUDA_DEBUG
	cout << "Version 0.1" << endl;
#endif
	args.GetReturnValue().Set(String::NewFromUtf8(args.GetIsolate(),"Version 0.1"));
}

void WebCUDA::GetDriverVersion(const v8::FunctionCallbackInfo<v8::Value>& args){
	int driverVersion = 0;
	cuDriverGetVersion(&driverVersion);
	cout << "Driver Version " << driverVersion << endl;
	args.GetReturnValue().Set(Integer::New(args.GetIsolate(), driverVersion));
}

void WebCUDA::GetDeviceCount(const v8::FunctionCallbackInfo<v8::Value>& args){
	int deviceCount = 0;
	cuDeviceGetCount(&deviceCount);
	cout << "Device Count " << deviceCount << endl;
	args.GetReturnValue().Set(Integer::New(args.GetIsolate(), deviceCount));
}

void WebCUDA::StartProfiling(const v8::FunctionCallbackInfo<v8::Value>& args){
	cuProfilerStart();
	cout << "Profiling Started!" << endl;
	args.GetReturnValue().Set(String::NewFromUtf8(args.GetIsolate(),"Profiling Started!"));
}

void WebCUDA::StopProfiling(const v8::FunctionCallbackInfo<v8::Value>& args){
	cuProfilerStop();
	cout << "Profiling Stopped!" << endl;
	args.GetReturnValue().Set(String::NewFromUtf8(args.GetIsolate(),"Profiling Stopped!"));
}
