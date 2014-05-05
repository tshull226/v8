void Device::MakeDeviceObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());
	//retrieving device number
	Handle<Integer> arg = Handle<Integer>::Cast(args[0]);
	int64_t deviceNum = arg->Value();
	//creating Device Object
	Device* pdevice = new Device();
	cuDeviceGet(&(pdevice->m_device), deviceNum);
	//creating Device Object Wrapper
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(args.GetIsolate(), constructor_template);
	Local<Object> result = templ->NewInstance();
	//wrap the raw C++ pointer in an External so it can be set as JavaScript Field
	Handle<External> device_ptr = External::New(args.GetIsolate(), pdevice);
	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, device_ptr);
	//set return value to Device Object Wrapper
	args.GetReturnValue().Set(result);
}
