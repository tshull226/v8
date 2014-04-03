#ifndef BINDINGS_H
#define BINDINGS_H

#include "../../../include/v8.h"
#include "webcuda_top.h"
//#include <node.h>

using namespace v8;
//using namespace v8::internal;
using v8::internal::AccessorInfo;

namespace WebCuda {

	/*
  static Handle<Value> GetDriverVersion(Local<String> property, const internal::AccessorInfo &info);
  static Handle<Value> GetDeviceCount(Local<String> property, const internal::AccessorInfo &info);
	*/
	
  static Handle<Value> GetDriverVersion(Local<String> property, const AccessorInfo &info);
  static Handle<Value> GetDeviceCount(Local<String> property, const AccessorInfo &info);
	

}

#endif
