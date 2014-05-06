void Device::MakeDeviceObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	% {\normalsize \bf // retrieving device number} *)
	Handle<Integer> arg = Handle<Integer>::Cast(args[0]);
	int64_t deviceNum = arg->Value();
	% {\normalsize \bf // creating Device Object} *)
	Device* pdevice = new Device();
	cuDeviceGet(&(pdevice->m_device), deviceNum);
	% {\normalsize \bf // creating Device Object Wrapper} *)
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(args.GetIsolate(), constructor_template);
	Local<Object> result = templ->NewInstance();
	% {\normalsize \bf // wrap the raw C++ pointer in an External so it can be set as JavaScript Field} *)
	Handle<External> device_ptr = External::New(args.GetIsolate(), pdevice);
	% {\normalsize \bf // Store the pointer in the JavaScript Wrapper} *)
	result->SetInternalField(0, device_ptr);
	% {\normalsize \bf // set return value to Device Object Wrapper} *)
	args.GetReturnValue().Set(result);
}
