#ifndef MODULE_HPP
#define MODULE_HPP

#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda {

  class Module {
    public:
			/**
			 * \brief initialize features of Module
			 */
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

			/**
			 * \brief extract specific function from a CUDA module
			 */
      static void  GetFunction(const v8::FunctionCallbackInfo<v8::Value>& args);

    protected:
      static Persistent<ObjectTemplate> constructor_template;

			/**
			 * \brief load a .cubin file into a JavaScript wrapper
			 */
      static void  Load(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief create JavaScript Object used to represent CUDA file
			 */
      static void  MakeModuleObject(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief retrieve C++ Module Object from JavaScript wrapper
			 */
			static Module* UnwrapModule(Handle<Object> obj);

      Module() : m_module(0) {}

      ~Module() {}

    private:
 //     static Handle<Value> New(const Arguments& args);
			
			/**
			 * \brief helper method for creating JavaScript Object used to represent CUDA file
			 */
      static Handle<Object>  MakeModuleObject_(Isolate* isolate);

			/**
			 * \brief creates object template for JavaScript Module Object
			 */
      static Handle<ObjectTemplate> MakeModuleTemplate(Isolate* isolate);

      CUmodule m_module;
  };

}

#endif
