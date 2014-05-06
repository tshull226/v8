#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cuda.h>
#include <v8.h>

using namespace v8;

namespace webcuda {

  class Function {
    public:
			/**
			 * \brief initialize features of Function
			 */
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

			/**
			 * \brief create JavaScript Object used to represent CUDA kernel
			 */
      static Handle<Object> MakeFunctionObject(Isolate* isolate) { return MakeFunctionObject_(isolate); }

			/**
			 * \brief retrieve C++ Function Object from JavaScript wrapper
			 */
      static Function* UnwrapFunction(Handle<Object> obj);
      CUfunction m_function;


    protected:
      static Persistent<ObjectTemplate> constructor_template;

			/**
			 * \brief create JavaScript Object used to represent CUDA kernel
			 */
      static void  MakeFunctionObject(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief launches CUDA kernel
			 */
      static void  LaunchKernel(const v8::FunctionCallbackInfo<v8::Value>& args);

      Function() : m_function(0) {}

      ~Function() {}

    private:
			
			/**
			 * \brief helper method for creating JavaScript Object used to represent CUDA kernel
			 */
      static Handle<Object> MakeFunctionObject_(Isolate* isolate);

			/**
			 * \brief creates object template for JavaScript Function Object
			 */
      static Handle<ObjectTemplate> MakeFunctionTemplate(Isolate* isolate);


  };

}

#endif
