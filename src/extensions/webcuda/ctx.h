#ifndef CTX_HPP
#define CTX_HPP

#include <cuda.h>
#include <v8.h>
//#include "device.h"

using namespace v8;

namespace webcuda{

	class Ctx {
		public:
			/**
			 * \brief initialize features of Function
			 */
			static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

		protected:
			static Persistent<ObjectTemplate> constructor_template;

			/**
			 * \brief create JavaScript Object used to represent CUDA Context
			 */
			static void  New(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief Delete CUDA context
			 */
			static void Destroy(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);

			/**
			 * \brief return CUDA Driver API version being used
			 */
			static void GetApiVersion(Local<String> name, const v8::PropertyCallbackInfo<v8::Value>& info);

			/**
			 * \brief stall until all CUDA operations are complete
			 */
			static void  Synchronize(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief push current CUDA context onto stack
			 */
			static void  PushCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief pop current CUDA context off stack
			 */
			static void  PopCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief set current CUDA context
			 */
			static void  SetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief retrieve current CUDA context
			 */
			static void  GetCurrent(const v8::FunctionCallbackInfo<v8::Value>& args);
			//static void  MakeCtxObject(const v8::FunctionCallbackInfo<v8::Value>& args);

			static Ctx* UnwrapCtx(Handle<Object> obj);

			Ctx() : m_context(0) {}
			//Ctx() : m_context(0), sync_in_progress(false) {}

			~Ctx () {}

		private:
			/**
			 * \brief helper method for creating JavaScript Object used to represent CUDA kernel
			 */
			static Handle<Object>  MakeCtxObject_(Isolate* isolate);

			/**
			 * \brief creates object template for JavaScript Ctx (Context) Object
			 */
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
