#include "ctx.h"
#include "device.h"

using namespace v8;
using namespace webcuda;

Persistent<ObjectTemplate> Ctx::constructor_template;

void Ctx::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);
/*
  Local<FunctionTemplate> t = FunctionTemplate::New(Ctx::New);
  constructor_template = Persistent<FunctionTemplate>::New(t);
  constructor_template->InstanceTemplate()->SetInternalFieldCount(1);
  constructor_template->SetClassName(String::NewSymbol("CudaCtx"));
	*/

	webcuda_templ->Set(String::NewFromUtf8(isolate, "newContext"),
			FunctionTemplate::New(isolate, Ctx::New));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "synchronize"),
			FunctionTemplate::New(isolate, Ctx::Synchronize));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "setCurrent"),
			FunctionTemplate::New(isolate, Ctx::SetCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "getCurrent"),
			FunctionTemplate::New(isolate, Ctx::GetCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "pushCurrent"),
			FunctionTemplate::New(isolate, Ctx::PushCurrent));

	webcuda_templ->Set(String::NewFromUtf8(isolate, "popCurrent"),
			FunctionTemplate::New(isolate, Ctx::PopCurrent));

	Handle<ObjectTemplate> raw_template = MakeCtxTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);
	/*
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "destroy", Ctx::Destroy);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "pushCurrent", Ctx::PushCurrent);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "popCurrent", Ctx::PopCurrent);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "setCurrent", Ctx::SetCurrent);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "getCurrent", Ctx::GetCurrent);
  NODE_SET_PROTOTYPE_METHOD(constructor_template, "synchronize", Ctx::Synchronize);
  constructor_template->InstanceTemplate()->SetAccessor(String::New("apiVersion"), Ctx::GetApiVersion);

  target->Set(String::NewSymbol("Ctx"), constructor_template->GetFunction());
	*/
}

Handle<ObjectTemplate> Ctx::MakeCtxTemplate(Isolate* isolate) {
  EscapableHandleScope handle_scope(isolate);

  Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
  result->SetInternalFieldCount(1);

  // Add accessors for each of the fields of the request.
	result->Set(String::NewFromUtf8(isolate, "destroy"),
			FunctionTemplate::New(isolate, Ctx::Destroy));
	/*
  result->SetAccessor(
      String::NewFromUtf8(isolate, "destroy", String::kInternalizedString),
      Ctx::Destroy);
			*/
  result->SetAccessor(
      String::NewFromUtf8(isolate, "apiVersion", String::kInternalizedString),
      Ctx::GetApiVersion);
	//TODO SHOULD DO THIS EVENTUALLY
	/*
  result->SetAccessor(
	String::NewFromUtf8(isolate, "device", String::kInternalizedString),
	Ctx::GetDevice);
	*/

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

	cuCtxCreate(&(pctx->m_context), flags, m_device);

	args.GetReturnValue().Set(ctx_ptr);
}

//void Ctx::Destroy(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
void Ctx::Destroy(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(args.Holder());

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
	//need to think about this a little more since currently creating a new context also sets it as the one running
	HandleScope scope(args.GetIsolate());
	Ctx *pctx = UnwrapCtx(Handle<Object>::Cast(args[0]));

	CUresult error = cuCtxPopCurrent(&(pctx->m_context));
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
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

	//CUresult error = cuCtxGetCurrent(&(pctx->m_context));
	//TODO NEED TO DO SOMETHING WITH ERROR
	cuCtxGetCurrent(&(pctx->m_context));
	args.GetReturnValue().Set(result);
}

//TODO: DON'T NEED THIS
/*
	 struct SynchronizeParams {
	 Ctx *ctx;
	 CUresult error;
	 Persistent<Function> cb;
	 };
	 */

//TODO Should not be a property method
void  Ctx::Synchronize(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope scope(args.GetIsolate());
	/*TODO: NEED TO LOOK MORE CAREFULLY INTO...
		if (args.Length() >= 1 && args[0]->IsFunction()) {
	// Asynchronous
	Local<Function> cb = Local<Function>::Cast(args[0]);

	Ctx *ctx = ObjectWrap::Unwrap<Ctx>(args.This());
	if (ctx->sync_in_progress)
	return scope.Close(Number::New(-1));

	SynchronizeParams *params = new SynchronizeParams();
	params->ctx = ctx;
	params->cb = Persistent<Function>::New(cb);

	cuCtxPopCurrent(NULL);

	// build up the work request
	uv_work_t* work_req = new uv_work_t;
	work_req->data = params;

	uv_queue_work(uv_default_loop(),
	work_req,
	Process,
	After);
	uv_ref((uv_handle_t*) &work_req);

	ctx->Ref();
	ctx->sync_in_progress = true;

	return Undefined();

	} else {
	// Synchronous
	CUresult error = cuCtxSynchronize();
	return scope.Close(Number::New(error));
	}
	*/
	CUresult error = cuCtxSynchronize();
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}



//TODO: DON'T THINK THAT I NEED ANY OF THIS...
/*
	 void Ctx::Process(uv_work_t* work_req) {
	 SynchronizeParams *params = static_cast<SynchronizeParams*>(work_req->data);

	 params->error = cuCtxPushCurrent(params->ctx->m_context);
	 if (params->error) return;

	 params->error = cuCtxSynchronize();
	 if (params->error) return;

	 params->error = cuCtxPopCurrent(NULL);
	 }

	 void Ctx::After(uv_work_t* work_req, int status) {
	 assert(status == 0);
	 HandleScope scope;
	 SynchronizeParams *params = static_cast<SynchronizeParams*>(work_req->data);

	 params->ctx->Unref();
	 params->ctx->sync_in_progress = false;

	 cuCtxPushCurrent(params->ctx->m_context);

	 Local<Value> argv[1];
	 argv[0] = Number::New(params->error);

	 TryCatch try_catch;
	 params->cb->Call(Context::GetCurrent()->Global(), 1, argv);
	 if (try_catch.HasCaught()) FatalException(try_catch);

	 params->cb.Dispose();
	 uv_unref((uv_handle_t*) work_req);
	 delete params;
	 }
	 */

void Ctx::GetApiVersion(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
	HandleScope scope(info.GetIsolate());
	Ctx* pctx = UnwrapCtx(info.Holder());

	unsigned int version;
	//CUresult error = cuCtxGetApiVersion(pctx->m_context, &version);
	//TODO NEED TO DO SOMETHING WITH ERROR
	cuCtxGetApiVersion(pctx->m_context, &version);


	info.GetReturnValue().Set(Number::New(info.GetIsolate(), version));
}

