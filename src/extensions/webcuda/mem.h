#ifndef MEM_HPP
#define MEM_HPP

#include <cuda.h>
#include <v8.h>
//#include "function.hpp" not sure why it would need this...
using namespace v8;

namespace webcuda {

  class Mem {
    public:
      static void Initialize(v8::Isolate* isolate, Handle<ObjectTemplate> webcuda_templ);

			/**
			 * \brief retrieves device pointer of device mem
			 */
      static void* GetDevicePtr(Handle<Object> obj, size_t* bufsize);

    protected:
			/**
			 * \brief initialize features of Mem
			 */
      static Persistent<ObjectTemplate> constructor_template;

			/**
			 * \brief create JavaScript Object used to represent CUDA memory
			 */
      static Handle<Object> MakeMemObject(Isolate* isolate);

			/**
			 * \brief creates object template for JavaScript Mem Object
			 */
      static Handle<ObjectTemplate> MakeMemTemplate(Isolate* isolate);

			/**
			 * \brief allocates CUDA memory for the application
			 */
      static void Alloc(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief allocates CUDA memory for the application
			 */
      static void AllocPitch(const v8::FunctionCallbackInfo<v8::Value>& args);
			
			/**
			 * \brief frees previously allocated CUDA memory
			 */
      static void Free(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief copies memory from JavaScript space to CUDA memory
			 */
      static void CopyHtoD(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief copies memory from CUDA memory to JavaScript space
			 */
      static void CopyDtoH(const v8::FunctionCallbackInfo<v8::Value>& args);

			/**
			 * \brief retrieve C++ Mem Object from JavaScript wrapper
			 */
			static Mem* UnwrapDevicePtr(Handle<Object> obj);

      //static Handle<Value> GetDevicePtr(Local<String> property, const AccessorInfo &info);

      Mem() : m_devicePtr(0) {}

      ~Mem() {}

    private:

      CUdeviceptr m_devicePtr;

//      friend class NodeCuda::Function; NEED TO SEE WHY THIS IS HERE...
  };

}

#endif
