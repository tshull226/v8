#ifndef CTX_HPP
#define CTX_HPP

#include <cuda.h>
#include <v8.h>
//#include "device.h"

using namespace v8;

namespace webcuda{

	class Ctx {
		public:
			static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

		protected:
			static Persistent<ObjectTemplate> constructor_template;

			static void  New(const v8::FunctionCallbackInfo<v8::Value>& args);

			static void Destroy(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);
			static void GetApiVersion(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);

			static void  Synchronize(const v8::FunctionCallbackInfo<v8::Value>& args);
			static void  PushCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
			static void  PopCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
			static void  SetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
			static void  GetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
			//static void  MakeCtxObject(const v8::FunctionCallbackInfo<v8::Value>& args);

			static Ctx* UnwrapCtx(Handle<Object> obj);

			Ctx() : m_context(0) {}
			//Ctx() : m_context(0), sync_in_progress(false) {}

			~Ctx () {}

		private:
			static Handle<Object>  MakeCtxObject_(Isolate* isolate);
			static Handle<ObjectTemplate> MakeCtxTemplate(Isolate* isolate);

			CUcontext m_context;

			/*
				 bool sync_in_progress;

				 uv_work_t work_req_;

				 static void Process(uv_work_t* work_req);
				 static void After(uv_work_t* work_req, int status);
				 */
	};

}

#endif
