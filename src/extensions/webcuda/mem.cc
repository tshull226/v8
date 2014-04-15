#include <cstring>
//#include <node_buffer.h> need to see what the implications of this is...
#include "mem.h"

using namespace v8;
using namespace webcuda;


Persistent<ObjectTemplate> Mem::constructor_template;

void Mem::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	Handle<ObjectTemplate> raw_template = MakeMemTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "memAlloc"),
			FunctionTemplate::New(isolate, Alloc));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "memAllocPitch"),
			FunctionTemplate::New(isolate, AllocPitch));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "free"),
			FunctionTemplate::New(isolate, Free));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "copyHtoD"),
			FunctionTemplate::New(isolate, CopyHtoD));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "copyDtoH"),
			FunctionTemplate::New(isolate, CopyDtoH));
}

Handle<ObjectTemplate> Mem::MakeMemTemplate(Isolate* isolate) {
  EscapableHandleScope handle_scope(isolate);

  Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
  result->SetInternalFieldCount(1);

	//Don't think I need any....
  // Add accessors for each of the fields of the request.
	/*
  result->SetAccessor(
      String::NewFromUtf8(isolate, "name", String::kInternalizedString),
      Device::GetName);
	*/

  // Again, return the result through the current handle scope.
  return handle_scope.Escape(result);
}

Handle<Object> Mem::MakeMemObject(v8::Isolate* isolate) {
	EscapableHandleScope handle_scope(isolate);

	//making mem object
	Mem* mdevice = new Mem();

	//creating object
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(isolate, constructor_template);
	Local<Object> result = templ->NewInstance();

	//wrap the raw C++ pointer in an External so it can be referenced from within JavaScript
	Handle<External> device_ptr = External::New(isolate, mdevice);

	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, device_ptr);

	return handle_scope.Escape(result);
}

Mem* Mem::UnwrapDevicePtr(Handle<Object> obj) {
	Handle<External> field = Handle<External>::Cast(obj->GetInternalField(0));
	void* ptr = field->Value();
	return static_cast<Mem*>(ptr);
}


/** \args the byte size of CUDA memory to allocate
 *
 * uses cuMemAlloc to allocate the amount of memory specified and returns it in
 * a JavaScript Wrapped Object
 */
void Mem::Alloc(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	Local<Object> result = MakeMemObject(args.GetIsolate());
  Mem *pmem = UnwrapDevicePtr(result);

  size_t bytesize = args[0]->Uint32Value();
	CUresult error = cuMemAlloc(&(pmem->m_devicePtr), bytesize);
	result->Set(String::NewFromUtf8(args.GetIsolate(), "size"), Integer::NewFromUnsigned(args.GetIsolate(), bytesize));
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}

/** \args the width, height, and byte size of CUDA memory array to allocate
 *
 * uses cuMemAllocPitch to allocate the amount of memory specified and returns it in
 * a JavaScript Wrapped Object which also tells of the size, pitch, and if an
 * error occured
 */
void Mem::AllocPitch(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	Local<Object> result = MakeMemObject(args.GetIsolate());
  Mem *pmem = UnwrapDevicePtr(result);

	size_t pPitch;
	unsigned int ElementSizeBytes = args[2]->Uint32Value();
	size_t WidthInBytes = ElementSizeBytes * args[0]->Uint32Value();
	size_t Height = args[1]->Uint32Value();
	CUresult error = cuMemAllocPitch(&(pmem->m_devicePtr), &pPitch, WidthInBytes, Height, ElementSizeBytes);

	result->Set(String::NewFromUtf8(args.GetIsolate(), "size"), Integer::NewFromUnsigned(args.GetIsolate(), pPitch*Height));
	result->Set(String::NewFromUtf8(args.GetIsolate(), "pitch"), Integer::NewFromUnsigned(args.GetIsolate(), pPitch));
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"), Integer::New(args.GetIsolate(), error));

	args.GetReturnValue().Set(result);
}


/** \args the JavaScript CUDA Memory object to be freed
 *
 * Uses cuMemFree to free the memory wrapped in the specific JavaScript Object
 */
void Mem::Free(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());

	Mem *pmem = UnwrapDevicePtr(Handle<Object>::Cast(args[0]));

	CUresult error = cuMemFree(pmem->m_devicePtr);

	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

/** \args contains both a JavaScript memory object as well as a JavaScript
 * ArrayBuffer
 *
 */
void Mem::CopyHtoD(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	Mem *pmem = UnwrapDevicePtr(Handle<Object>::Cast(args[0]));

	/*Local<Object> buf = args[1]->ToObject();
	//char *phost = Buffer::Data(buf);
  char *phost = static_cast<char*>(buf->GetIndexedPropertiesExternalArrayData());
	//size_t bytes = Buffer::Length(buf);
  size_t bytes=  buf->GetIndexedPropertiesExternalArrayDataLength();
	*/
	Handle<ArrayBuffer> buf = Handle<ArrayBuffer>::Cast(args[1]);
	v8::ArrayBuffer::Contents ctx = buf->Externalize();
  //char *phost = static_cast<char*>(ctx.Data());
  void *phost = ctx.Data();
	size_t bytes = ctx.ByteLength();

	//bool async = args.Length() >= 2 && args[2]->IsTrue();

	CUresult error;
	error = cuMemcpyHtoD(pmem->m_devicePtr, phost, bytes);
	/*
	if (async) {
		error = cuMemcpyHtoDAsync(pmem->m_devicePtr, phost, bytes, 0);
	} else {
		error = cuMemcpyHtoD(pmem->m_devicePtr, phost, bytes);
	}
	*/

	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

void Mem::CopyDtoH(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());

	Handle<ArrayBuffer> buf = Handle<ArrayBuffer>::Cast(args[0]);
	v8::ArrayBuffer::Contents ctx = buf->Externalize();
  //char *phost = static_cast<char*>(ctx.Data());
  void *phost = ctx.Data();
	size_t bytes = ctx.ByteLength();
	/*
	Local<Object> buf = args[0]->ToObject();
	//char *phost = Buffer::Data(buf);
  char *phost = static_cast<char*>(buf->GetIndexedPropertiesExternalArrayData());
	//size_t bytes = Buffer::Length(buf);
  size_t bytes=  buf->GetIndexedPropertiesExternalArrayDataLength();
	*/

	Mem *pmem = UnwrapDevicePtr(Handle<Object>::Cast(args[1]));

	//bool async = args.Length() >= 2 && args[2]->IsTrue();

	CUresult error;
	error = cuMemcpyDtoH(phost, pmem->m_devicePtr, bytes);
	/*
	if (async) {
		error = cuMemcpyDtoHAsync(pmem->m_devicePtr, phost, bytes, 0);
	} else {
		error = cuMemcpyDtoH(pmem->m_devicePtr, phost, bytes);
	}
	*/

	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

//Not sure why this exists...
/*
Handle<Value> Mem::GetDevicePtr(Local<String> property, const AccessorInfo &info) {
	HandleScope scope;
	Mem *pmem = ObjectWrap::Unwrap<Mem>(info.Holder());
	Buffer *ptrbuf = Buffer::New(sizeof(pmem->m_devicePtr));

	memcpy(Buffer::Data(ptrbuf->handle_), &pmem->m_devicePtr, sizeof(pmem->m_devicePtr));

	return scope.Close(ptrbuf->handle_);
}
*/
