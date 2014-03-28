
#include <v8.h>

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
