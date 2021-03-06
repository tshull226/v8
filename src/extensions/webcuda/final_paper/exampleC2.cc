void Mem::Free(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	% {\normalsize \bf // retrieve pointer to Memory device} *)
	Mem *pmem = UnwrapDevicePtr(Handle<Object>::Cast(args[0]));
	% {\normalsize \bf // perform CUDA Driver call} *)
	CUresult error = cuMemFree(pmem->m_devicePtr);
	% {\normalsize \bf // return function result} *)
	args.GetReturnValue().Set(Number::New(args.GetIsolate(), error));
}
