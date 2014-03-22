
#include <iostream>
#include "webcuda.h"

using namespace webcuda;
using namespace v8;
using std::cout;
using std::endl;

void WebCUDA::AddWebCUDAMethods(Isolate* isolate, Handle<ObjectTemplate> webcuda_templ){
	//doing nothing for the time being
	webcuda_templ->Set(String::NewFromUtf8(isolate, "version"),
			FunctionTemplate::New(isolate, Version));
}

void WebCUDA::Version(const v8::FunctionCallbackInfo<v8::Value>& args){
	cout << "Version 0.1" << endl;
	args.GetReturnValue().Set(String::NewFromUtf8(args.GetIsolate(),"Version 0.1"));
}
