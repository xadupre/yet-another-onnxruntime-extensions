// Source: https://github.com/microsoft/onnxruntime/tree/main/
// onnxruntime/test/testdata/custom_op_get_const_input_test_library
// Adapted from https://github.com/sdpython/onnx-extended

#include <mutex>
#include <vector>

#include "ort_optim_cpu_lib.h"
#include "ort_sparse.hpp"
#include "ortapi_version.h"

static const char *c_OpDomain = "yaourt.ortops.optim.cpu";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION_SUPPORTED));
  Ort::UnownedSessionOptions session_options(options);

  // Instances remaining available until onnxruntime unloads the library.
  static ortops::DenseToSparse<float> c_DenseToSparse;
  static ortops::SparseToDense<float> c_SparseToDense;

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(&c_DenseToSparse);
    domain.Add(&c_SparseToDense);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}
