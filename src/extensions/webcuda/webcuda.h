
#include <v8.h>

/** \mainpage WebCUDA Documentation
 *
 * \section intro_sec Introduction
 *
 * These webpages consist of documentation for WebCUDA's implementation in v8.
 * The goal of WebCUDA is to add CUDA bindings to JavaScript. This project was started as
 * as class project for CS598SVA at the University of Illinois in Urbana-Champaign.
 *
 * \section webcuda_spec WebCUDA Specification
 * The WebCUDA specification can be found <a href="http://tshull226.synology.me/CS598SVA/JSDoc/index.html">here</a>
 *
 * \section webcuda_code WebCUDA Code
 * The code used as the basis for this documentation can be found
 * <a href="https://github.com/tshull226/v8">here</a>
 *
 * \subsection Contributors
 * Jiho Choi <br/>
 * Kurt Fellows <br/>
 * Tom Shull
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
