
#include <v8.h>

/** \mainpage WebCUDA Documentation
 *
 * These webpages consist of the WebCUDA documentation. The goal of WebCUDA is
 * to add CUDA bindings for JavaScript for V8 in the Chrome and d8 applications.
 * This is the CS598 SVA project for Jiho Choi, Kurt Fellows, and Tom Shull. We
 * base this project on node-cuda, a similar project for Node.js.
 * 
 * Stay Tuned for future updates!
 */

using namespace v8;

namespace webcuda{
	class WebCUDA{
		public:
			/**
			 * \brief returns webCUDA Version Number
			 */

			static void Version(const v8::FunctionCallbackInfo<v8::Value>& args);
			/**
			 * \brief returns CUDA driver version value
			 */
			static void GetDriverVersion(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief returns number of CUDA drivers available
			 */
			static void GetDeviceCount(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief signals NVIDIA Profiling Tools to Start
			 */
			static void StartProfiling(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief signals NVIDIA Profiling Tools to Stop
			 */
			static void StopProfiling(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief initialize and add webCUDA features into the v8 object enviroment
			 */
			static void AddWebCUDAMethods(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);
	};
}
