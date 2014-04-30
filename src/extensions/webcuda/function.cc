//#include <node_buffer.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include "function.h"
#include "mem.h"

using namespace webcuda;
using namespace v8;
using std::cout;
using std::endl;




Persistent<ObjectTemplate> webcuda::Function::constructor_template;

void webcuda::Function::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
	HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "launchKernel"),
			FunctionTemplate::New(isolate, LaunchKernel));

	Handle<ObjectTemplate> raw_template = MakeFunctionTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

	//target->Set(String::NewSymbol("Device"), constructor_template->GetFunction());
}

Handle<ObjectTemplate> webcuda::Function::MakeFunctionTemplate(Isolate* isolate) {
	EscapableHandleScope handle_scope(isolate);

	Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
	result->SetInternalFieldCount(1);

	// Again, return the result through the current handle scope.
	return handle_scope.Escape(result);
}

//void MakeFunctionObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
Handle<Object> webcuda::Function::MakeFunctionObject_(Isolate* isolate) {
	EscapableHandleScope handle_scope(isolate);

	//retrieving device information

	Function* fdevice = new Function();

	//creating object
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(isolate, constructor_template);
	Local<Object> result = templ->NewInstance();

	//wrap the raw C++ pointer in an External so it can be referenced from within JavaScript
	Handle<External> device_ptr = External::New(isolate, fdevice);

	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, device_ptr);

	return handle_scope.Escape(result);
}

webcuda::Function* webcuda::Function::UnwrapFunction(Handle<Object> obj) {
	Handle<External> field = Handle<External>::Cast(obj->GetInternalField(0));
	void* ptr = field->Value();
	return static_cast<Function*>(ptr);
}

void webcuda::Function::MakeFunctionObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	Handle<Object> temp = MakeFunctionObject_(args.GetIsolate());

	args.GetReturnValue().Set(temp);
}

/** \param args contains the block dimensions, thread dimensions, and the arguments for the kernel launch
 *
 * Function uses cuLaunchKernel)
 */
void webcuda::Function::LaunchKernel(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	Function* pfunction = UnwrapFunction(Handle<Object>::Cast(args[0]));
	/*
	//testing function
	CUfunction *func = &(pfunction->m_function);
	int test;
	CUresult e1 = cuFuncGetAttribute(&test, CU_FUNC_ATTRIBUTE_NUM_REGS, *func);
	cout << "test error:" << e1 << ", result:" << test << endl;
	e1 = cuFuncGetAttribute(&test, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, *func);
	cout << "test error:" << e1 << ", result:" << test << endl;
	*/

	Local<Array> gridDim = Local<Array>::Cast(args[1]);
	unsigned int gridDimX = gridDim->Get(0)->Uint32Value();
	unsigned int gridDimY = gridDim->Get(1)->Uint32Value();
	unsigned int gridDimZ = gridDim->Get(2)->Uint32Value();
	cout << gridDimX << "," << gridDimY << "," << gridDimZ << endl;

	Local<Array> blockDim = Local<Array>::Cast(args[2]);
	unsigned int blockDimX = blockDim->Get(0)->Uint32Value();
	unsigned int blockDimY = blockDim->Get(1)->Uint32Value();
	unsigned int blockDimZ = blockDim->Get(2)->Uint32Value();

	/*
		 Local<Object> buf = args[3]->ToObject();
	//char *pbuffer = Buffer::Data(buf);
	char *pbuffer = static_cast<char*>(buf->GetIndexedPropertiesExternalArrayData());
	//size_t bufferSize = Buffer::Length(buf);
	size_t bufferSize =  buf->GetIndexedPropertiesExternalArrayDataLength();
	*/


	void **kernelParams = NULL;
	void **pointerVals = NULL;
	size_t ptrValLen = 0;
	size_t sharedMemory = 0;
	//Handle<Value> temp = obj->GetRealNamedProperty(String::NewFromUtf8(args.GetIsolate(), "devicePtr"));
	Handle<Value> temp = args[3];
	Handle<Integer> val = Handle<Integer>::Cast(temp);
	sharedMemory = val->Value();


	Local<Array> argArray = Local<Array>::Cast(args[4]);
	size_t len = argArray->Length();
	if(len > 0){
		kernelParams = (void **) malloc(sizeof(void *)*len);
		pointerVals = (void **) malloc(sizeof(void *)*len);
		Handle<String> strHandle = String::NewFromUtf8(args.GetIsolate(), "memParam");
		Handle<String> intHandle = String::NewFromUtf8(args.GetIsolate(), "intParam");
		Handle<String> floatHandle = String::NewFromUtf8(args.GetIsolate(), "floatParam");
		Handle<String> doubleHandle = String::NewFromUtf8(args.GetIsolate(), "doubleParam");
		for(size_t i = 0; i < len; i++){
			Handle<Object> obj = Handle<Object>::Cast(argArray->Get(i));
			if(obj->HasOwnProperty(strHandle)){
				//Handle<Value> temp = obj->GetRealNamedProperty(String::NewFromUtf8(args.GetIsolate(), "devicePtr"));
				Handle<Value> temp = obj->Get(strHandle);
				Handle<Object> mem = Handle<Object>::Cast(temp);
				size_t bufferSize;
				void *bufferLoc = Mem::GetDevicePtr(mem, &bufferSize);
				kernelParams[i] = bufferLoc;
			} else if(obj->HasOwnProperty(intHandle)){
				//Handle<Value> temp = obj->GetRealNamedProperty(String::NewFromUtf8(args.GetIsolate(), "devicePtr"));
				Handle<Value> temp = obj->Get(intHandle);
				Handle<Integer> val = Handle<Integer>::Cast(temp);
				int *intVal = (int *)malloc(sizeof(int));
				*intVal = val->Value();
				kernelParams[i] = intVal;
				pointerVals[ptrValLen] = intVal;
				ptrValLen++;
			} else if(obj->HasOwnProperty(floatHandle)){
				//Handle<Value> temp = obj->GetRealNamedProperty(String::NewFromUtf8(args.GetIsolate(), "devicePtr"));
				Handle<Value> temp = obj->Get(floatHandle);
				float *floatVal = (float *)malloc(sizeof(float));
				*floatVal = temp->NumberValue();
				kernelParams[i] = floatVal;
				pointerVals[ptrValLen] = floatVal;
				ptrValLen++;
			} else if(obj->HasOwnProperty(doubleHandle)){
				//Handle<Value> temp = obj->GetRealNamedProperty(String::NewFromUtf8(args.GetIsolate(), "devicePtr"));
				Handle<Value> temp = obj->Get(floatHandle);
				float *floatVal = (float *)malloc(sizeof(float));
				*floatVal = temp->NumberValue();
				kernelParams[i] = floatVal;
				pointerVals[ptrValLen] = floatVal;
				ptrValLen++;
			} else {
				//currenty cannot handle the parameters given

				//need to free any memory allocated
				for(size_t i = 0; i < ptrValLen; i++){
					free(pointerVals[i]);
				}
				free(pointerVals);
				free(kernelParams);

				args.GetReturnValue().Set(Number::New(args.GetIsolate(), -1));
			}
		}
	} 

#ifdef V8_WEBCUDA_DEBUG
	cout << "sharedMemory: " << sharedMemory << endl;
	cout << "num ptrs " << ptrValLen << endl;
#endif

	CUresult error = cuLaunchKernel(pfunction->m_function,
			gridDimX, gridDimY, gridDimZ,
			blockDimX, blockDimY, blockDimZ,
			sharedMemory, 0, kernelParams, NULL);

	//Freeing memory that possibly was created

	if(len > 0){
		for(size_t i = 0; i < ptrValLen; i++){
			free(pointerVals[i]);
		}
		free(pointerVals);
		free(kernelParams);
	}


#ifdef V8_WEBCUDA_DEBUG
	if(error == CUDA_ERROR_INVALID_VALUE){
		cout << "invalid parameters send to kernel Launch" << endl;
	}
#endif

	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

