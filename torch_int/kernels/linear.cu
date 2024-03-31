#include "include/linear.h"
#include "include/common.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>


#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/gemm/device/gemm_universal_streamk_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

// used by in_proj
torch::Tensor linear_with_token_scaling(
  torch::Tensor input,  // FP32 (M=bsize*L, K=in_feat)
  torch::Tensor weight, // FP32 (N=out_feat, K=in_feat)
  torch::Tensor bias,   // FP32
  torch::Tensor alpha  // FP32
) {

  // set M, N, K
  int M = static_cast<int>(input.size(0));  // input: (M, K)
  int N = static_cast<int>(weight.size(0)); // weight: (N, K)
  int K = static_cast<int>(input.size(1)); 
  printf("M, N, K: %d, %d, %d\n", M, N, K);

  // use the broadcasted bias as the output
  auto device = input.device();
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
  /////////////////////////////////////////////////////////////////////////////////////////////////

  using CUType = cutlass::half_t;

  CUType* alpha_ref = reinterpret_cast<CUType *>(alpha.data_ptr());

  // A matrix configuration
  using         ElementA         = CUType;                                  // Element type for A matrix operand
  using         LayoutA          = cutlass::layout::RowMajor;                        // Layout type for A matrix operand
  constexpr int AlignmentA       = 128 / cutlass::sizeof_bits<ElementA>::value;      // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
  ElementA* input_ref = reinterpret_cast<ElementA *>(input.data_ptr());

  // B matrix configuration
  using         ElementB         = CUType;                                  // Element type for B matrix operand
  using         LayoutB          = cutlass::layout::ColumnMajor;                        // Layout type for B matrix operand
  constexpr int AlignmentB       = 128 / cutlass::sizeof_bits<ElementB>::value;      // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
  ElementB* weight_ref = reinterpret_cast<ElementB *>(weight.data_ptr());

  // Output matrix configuration
  using         ElementOutput    = CUType;                                  // Element type for output matrix operands
  using         LayoutOutput     = cutlass::layout::RowMajor;                        // Layout type for output matrix operands
  constexpr int AlignmentOutput  = 128 / cutlass::sizeof_bits<ElementOutput>::value; // Memory access granularity/alignment of output matrices in units of elements (up to 16 bytes)
  ElementOutput* out_ref = reinterpret_cast<ElementOutput *>(out.data_ptr());

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator  = CUType;                          // Element type for internal accumulation
  using ElementCompute      = CUType;                          // Element type for compute
  using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
  // using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 32>;   // Threadblock-level tile size (concept: GemmShape)
  // using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;     // Warp-level tile size (concept: GemmShape)
  using ThreadblockShape    = cutlass::gemm::GemmShape<16, 64, 64>;   // Threadblock-level tile size (concept: GemmShape)
  using WarpShape           = cutlass::gemm::GemmShape<16, 64, 64>;     // Warp-level tile size (concept: GemmShape)
  // using ThreadblockShape    = cutlass::gemm::GemmShape<16, 32, 32>;   // Threadblock-level tile size (concept: GemmShape)
  // using WarpShape           = cutlass::gemm::GemmShape<16, 32, 32>;     // Warp-level tile size (concept: GemmShape)
  using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;      // Instruction-level tile size (concept: GemmShape)
  constexpr int NumStages   = 2;                                        // Number of global->shared pipeline stages used in the GEMM mainloop
  constexpr int EVTEpilogueStages = 1;                                  // Number of epilogue stages in EVT

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<  
      ElementOutput,                        ///< Data type used to load and store tensors
      128 / cutlass::sizeof_bits<ElementOutput>::value,                                    ///< Number of elements computed per operation.
                                            ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                            ///< but we use 64 or 32 sometimes when there are not enough data to store
      ElementAccumulator,                   ///< Accumulator data type
      ElementCompute                        ///< Data type used to compute linear combination
      >;

  // StreamK device GEMM implementation type with EVT
  using namespace cute;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, 
    WarpShape, 
    ElementOutput, 
    AlignmentOutput, 
    EVTEpilogueStages
  >;

  // Accumulator
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  
  // alpha
  using Alpha = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, ElementOutput,
      cute::Stride<_1, _0, int32_t>
  >;
  // mul
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  // alpha * accumulator
  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
      Compute0, Alpha, Accum>;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, _1, int64_t> // StrideMNL
  >;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
      D,
      EVTCompute0>;

  using EVTKernelStreamK =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
      ElementOutput, LayoutOutput, AlignmentOutput,
      ElementAccumulator,
      ElementCompute,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      EVTEpilogueStages
  >::GemmKernel;

  using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<EVTKernelStreamK>;


  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmStreamK device_gemm;


  typename EVTD::Arguments callback_args{
    {
      {alpha_ref, ElementOutput(0), {_1{}, _0{}, int32_t(M)}},                 // Bias
      {},                                                                                                          // Accum
      {}                                                                                                           // Compute0
    },                                                                                                             // EVTCompute0
    {out_ref, {N, _1{}, M*N}},                   // D
  };    

  DeviceGemmStreamK::Arguments arguments (
      cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
      { M, N, K },                     // problem_size
      1,                   // batch count / splitk slices
      callback_args,                            // argument of EVT callbacks
      input_ref,                                        // ptr_A
      weight_ref,                                        // ptr_B
      nullptr,                                  // ptr_C (unused)
      nullptr,                                  // ptr_D (unused)
      M*K,      // batch_stride_A
      N*K,      // batch_stride_B
      0,                                        // batch_stride_C (unused)
      0,                                        // batch_stride_D (unused)
      K,              // stride_a
      K,              // stride_b
      0,                                        // stride_c (unused)
      0,                                        // stride_d (unused)
      -1                         // avail_sms
  );

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmStreamK::get_workspace_size(arguments);
  printf("workspace_size: %d\n", workspace_size);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

  // Execute
  CUTLASS_CHECK(device_gemm(arguments));

  return out;
}

// used by fc1
torch::Tensor linear_relu_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
                                      torch::Tensor bias,   // INT8
                                      float alpha,          // FP32
                                      float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3>;
#elif CUDA_ARCH >= 750
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      EpilogueOp>;
#elif CUDA_ARCH >= 700
  // LinearCombinationRelu does work with sm70, so we use torch relu instead.
  #define USE_TORCH_RELU
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>;
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      EpilogueOp>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);
  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int8_t>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement, status: " +
                             std::to_string((int)status));
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize, status: " +
                             std::to_string((int)status));
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run, status: " +
                             std::to_string((int)status));
  }

#ifdef USE_TORCH_RELU
#undef USE_TORCH_RELU
  out = torch::relu(out);
#endif

  return out;
}
