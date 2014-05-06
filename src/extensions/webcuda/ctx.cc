#include "ctx.h"
#include "device.h"

using namespace v8;
using namespace webcuda;

Persistent<ObjectTemplate> Ctx::constructor_template;

void Ctx::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "Context"),
			FunctionTemplate::New(isolate, Ctx::New));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "synchronizeCtx"),
			FunctionTemplate::New(isolate, Ctx::Synchronize));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "setCurrentCtx"),
			FunctionTemplate::New(isolate, Ctx::SetCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "getCurrentCtx"),
			FunctionTemplate::New(isolate, Ctx::GetCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "pushCurrentCtx"),
			FunctionTemplate::New(isolate, Ctx::PushCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "popCurrentCtx"),
			FunctionTemplate::New(isolate, Ctx::PopCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "destroyCtx"),
			FunctionTemplate::New(isolate, Ctx::Destroy));

	Handle<ObjectTemplate> raw_template = MakeCtxTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);
}

Handle<ObjectTemplate> Ctx::MakeCtxTemplate(Isolate* isolate) {
  EscapableHandleScope handle_scope(isolate);

  Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
  result->SetInternalFieldCount(1);

  // Add accessors for each of the fields of the request.
  result->SetAccessor(
      String::NewFromUtf8(isolate, "apiVersion", String::kInternalizedString),
      Ctx::GetApiVersion);
	// Again, return the result through the current handle scope.
	return handle_scope.Escape(result);
}

Handle<Object> Ctx::MakeCtxObject_(Isolate* isolate) {
	EscapableHandleScope handle_scope(isolate);

	//retrieving device information

	Ctx* cdevice = new Ctx();
	//cuDeviceGet(&(pdevice->m_device), deviceNum);

	//creating object
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(isolate, constructor_template);
	Local<Object> result = templ->NewInstance();

	//wrap the raw C++ pointer in an External so it can be referenced from within JavaScript
	Handle<External> device_ptr = External::New(isolate, cdevice);

	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, device_ptr);

	return handle_scope.Escape(result);
}

Ctx* Ctx::UnwrapCtx(Handle<Object> obj) {
	Handle<External> field = Handle<External>::Cast(obj->GetInternalField(0));
	void* ptr = field->Value();
	return static_cast<Ctx*>(ptr);
}

void Ctx::New(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());

	Handle<Object> ctx_ptr = MakeCtxObject_(args.GetIsolate());
	Ctx* pctx = UnwrapCtx(ctx_ptr);

	unsigned int flags = args[0]->Uint32Value();
	//pctx->m_device = ObjectWrap::Unwrap<Device>(args[1]->ToObject())->m_device;
	CUdevice m_device = Device::UnwrapDevice(Handle<Object>::Cast(args[1]))->m_device;

	CUresult error = cuCtxCreate(&(pctx->m_context), flags, m_device);
	ctx_ptr->Set(String::NewFromUtf8(args.GetIsolate(), "error"),
			Number::New(args.GetIsolate(), error));
	args.GetReturnValue().Set(ctx_ptr);
}

void Ctx::Destroy(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(Handle<Object>::Cast(args[0]));

	CUresult error = cuCtxDestroy(pctx->m_context);
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

void  Ctx::PushCurrent(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(Handle<Object>::Cast(args[0]));

	CUresult error = cuCtxPushCurrent(pctx->m_context);
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

void  Ctx::PopCurrent(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Handle<Object> result = MakeCtxObject_(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(result);

	CUresult error = cuCtxPopCurrent(&(pctx->m_context));
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"),
			Number::New(args.GetIsolate(), error));
	args.GetReturnValue().Set(result);
}

void Ctx::SetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(Handle<Object>::Cast(args[0]));

	CUresult error = cuCtxSetCurrent(pctx->m_context);
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}

void Ctx::GetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Handle<Object> result = MakeCtxObject_(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(result);

	CUresult error = cuCtxGetCurrent(&(pctx->m_context));
	result->Set(String::NewFromUtf8(args.GetIsolate(), "error"),
			Number::New(args.GetIsolate(), error));
	args.GetReturnValue().Set(result);
}


void  Ctx::Synchronize(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	CUresult error = cuCtxSynchronize();
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}




void Ctx::GetApiVersion(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
	HandleScope scope(info.GetIsolate());
	Ctx* pctx = UnwrapCtx(info.Holder());

	unsigned int version;
	//CUresult error = cuCtxGetApiVersion(pctx->m_context, &version);
	//TODO NEED TO DO SOMETHING WITH ERROR
	cuCtxGetApiVersion(pctx->m_context, &version);


	info.GetReturnValue().Set(Number::New(info.GetIsolate(), version));
}

