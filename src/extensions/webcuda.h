
#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda{
	class WebCUDA{
		public:
			CUmodule module;
			static void Version(const v8::FunctionCallbackInfo<v8::Value>& args);
			
			static void AddWebCUDAMethods(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);
	};
}
