#include "object_wrap.h"
#include "../../../include/v8.h"

#ifndef WEBCUDA_H
#define WEBCUDA_H

namespace V8_WebCuda{

	template <typename TypeName>
		inline void WEBCUDA_SET_METHOD(const TypeName& recv,
				const char* name,
				v8::FunctionCallback callback) {
			v8::Isolate* isolate = v8::Isolate::GetCurrent();
			v8::HandleScope handle_scope(isolate);
			v8::Local<v8::FunctionTemplate> t = v8::FunctionTemplate::New(isolate,
					callback);
			v8::Local<v8::Function> fn = t->GetFunction();
			v8::Local<v8::String> fn_name = v8::String::NewFromUtf8(isolate, name);
			fn->SetName(fn_name);
			recv->Set(fn_name, fn);
		}
#define WEBCUDA_SET_METHOD V8_WebCuda::WEBCUDA_SET_METHOD

	// Used to be a macro, hence the uppercase name.
	// Not a template because it only makes sense for FunctionTemplates.
	inline void WEBCUDA_SET_PROTOTYPE_METHOD(v8::Handle<v8::FunctionTemplate> recv,
			const char* name,
			v8::FunctionCallback callback) {
		v8::Isolate* isolate = v8::Isolate::GetCurrent();
		v8::HandleScope handle_scope(isolate);
		v8::Local<v8::FunctionTemplate> t = v8::FunctionTemplate::New(isolate,
				callback);
		recv->PrototypeTemplate()->Set(v8::String::NewFromUtf8(isolate, name),
				t->GetFunction());
	}

#define WEBCUDA_SET_PROTOTYPE_METHOD V8_WebCuda::WEBCUDA_SET_PROTOTYPE_METHOD


}

#endif
