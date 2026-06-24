#include <memory>
#include <mutex>
#include <vector>

#include "addadd_cpu.hpp"
#include "cpu_features.h"
#include "mulmul_cpu.hpp"
#include "ort_fused_kernel_cpu_lib.h"
#include "ortapi_version.h"

static const char *c_OpDomain = "yaourt.ortops.fused_kernel.cpu";

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
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_AddAddFloat{
      Ort::Custom::CreateLiteCustomOp<ortops::AddAddKernelCpuFloat>("AddAdd",
                                                                     "CPUExecutionProvider")};
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_AddAddFloat16{
      Ort::Custom::CreateLiteCustomOp<ortops::AddAddKernelCpuFloat16>("AddAdd",
                                                                       "CPUExecutionProvider")};
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_AddAddBFloat16{
      Ort::Custom::CreateLiteCustomOp<ortops::AddAddKernelCpuBFloat16>("AddAdd",
                                                                        "CPUExecutionProvider")};
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> c_MulMul{
      Ort::Custom::CreateLiteCustomOp<ortops::MulMulKernelCpu>("MulMul",
                                                               "CPUExecutionProvider")};

  try {
    Ort::CustomOpDomain domain{c_OpDomain};

    domain.Add(c_AddAddFloat.get());
    domain.Add(c_AddAddFloat16.get());
    domain.Add(c_AddAddBFloat16.get());
    domain.Add(c_MulMul.get());

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    return status.release();
  }

  return nullptr;
}

bool ORT_API_CALL CpuSupportsAvx2() { return ortops::cpu_supports_avx2(); }

bool ORT_API_CALL CpuSupportsAvx512f() { return ortops::cpu_supports_avx512f(); }

bool ORT_API_CALL CpuSupportsF16c() { return ortops::cpu_supports_f16c(); }
