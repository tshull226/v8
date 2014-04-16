#include "device.h"
#include <string>
#include <iostream>

using namespace webcuda;
using namespace v8;
using std::cout;
using std::endl;

Persistent<ObjectTemplate> Device::constructor_template;

/**
 * adds method "Device(int)" to webcuda object which returns an object containing information about the queried device
 */
void Device::Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ) {
  HandleScope scope(isolate);

	webcuda_templ->Set(String::NewFromUtf8(isolate, "Device"),
			FunctionTemplate::New(isolate, MakeDeviceObject));

	Handle<ObjectTemplate> raw_template = MakeDeviceTemplate(isolate);
	constructor_template.Reset(isolate, raw_template);

  //target->Set(String::NewSymbol("Device"), constructor_template->GetFunction());
}

Handle<ObjectTemplate> Device::MakeDeviceTemplate(Isolate* isolate) {
  EscapableHandleScope handle_scope(isolate);

  Local<ObjectTemplate> result = ObjectTemplate::New(isolate);
  result->SetInternalFieldCount(1);

  // Add accessors for each of the fields of the request.
  result->SetAccessor(
      String::NewFromUtf8(isolate, "name", String::kInternalizedString),
      Device::GetName);
  result->SetAccessor(
      String::NewFromUtf8(isolate, "totalMem", String::kInternalizedString),
      Device::GetTotalMem);
  result->SetAccessor(
      String::NewFromUtf8(isolate, "computeCapability", String::kInternalizedString),
      Device::GetComputeCapability);

  // Again, return the result through the current handle scope.
  return handle_scope.Escape(result);
}

void Device::MakeDeviceObject(const v8::FunctionCallbackInfo<v8::Value>& args) {
	HandleScope handle_scope(args.GetIsolate());

	//retrieving device information
	Handle<Integer> arg = Handle<Integer>::Cast(args[0]);
	int64_t deviceNum = arg->Value();
#ifdef V8_WEBCUDA_DEBUG
	cout << deviceNum << endl;
#endif

	
	Device* pdevice = new Device();
	cuDeviceGet(&(pdevice->m_device), deviceNum);

	//creating object
	Handle<ObjectTemplate> templ = Local<ObjectTemplate>::New(args.GetIsolate(), constructor_template);
	Local<Object> result = templ->NewInstance();

	//wrap the raw C++ pointer in an External so it can be referenced from within JavaScript
	Handle<External> device_ptr = External::New(args.GetIsolate(), pdevice);

	//Store the pointer in the JavaScript Wrapper
	result->SetInternalField(0, device_ptr);

	args.GetReturnValue().Set(result);
	
}


Device* Device::UnwrapDevice(Handle<Object> obj) {
	Handle<External> field = Handle<External>::Cast(obj->GetInternalField(0));
	void* ptr = field->Value();
	return static_cast<Device*>(ptr);
}

void Device::GetName(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
	Device* device = UnwrapDevice(info.Holder());

	char deviceName[256];
	cuDeviceGetName(deviceName, 256, device->m_device);
	std::string deviceStr(deviceName);

	info.GetReturnValue().Set(String::NewFromUtf8(info.GetIsolate(), deviceStr.c_str()));
}

void Device::GetComputeCapability(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
	Device* device = UnwrapDevice(info.Holder());

	int major = 0, minor = 0;
	cuDeviceComputeCapability(&major, &minor, device->m_device);

	Local<Object> result = Object::New(info.GetIsolate());
	result->Set(String::NewFromUtf8(info.GetIsolate(), "major"), Integer::New(info.GetIsolate(), major));
	result->Set(String::NewFromUtf8(info.GetIsolate(), "minor"), Integer::New(info.GetIsolate(), minor));

	info.GetReturnValue().Set(result);
}

void Device::GetTotalMem(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info) {
	Device* device = UnwrapDevice(info.Holder());

	size_t totalGlobalMem;
	cuDeviceTotalMem(&totalGlobalMem, device->m_device);

	info.GetReturnValue().Set(Integer::New(info.GetIsolate(), totalGlobalMem));
}
