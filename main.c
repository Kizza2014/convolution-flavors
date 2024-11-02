// @file main.c
//
//  \date Created on: Sep 22, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include "./inc/conv_layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "./inc/common_types.h"
#include "./inc/data_reshape.h"
#include "./inc/utils.h"
#include <cblas.h>
#include <openblas_config.h>
#include <sys/time.h>

struct timeval start, end;
long long time_taken;

static const int conv_test_k = 3;
static const int conv_test_in_c = 3;
static const int conv_test_in_h = 5;
static const int conv_test_in_w = 5;
static const int conv_test_out_c = 2;
static const int conv_test_batch = 1;

static const float conv_test_in_data[] = {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
    3, 3, 3, 3, 3,
};

static const float conv_test_filter[] = {
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
    1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9
};

static const float conv_test_bias[] = {
    0, 0
};

static const float conv_test_ref_out[] = {
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    /*------------------------------------*/
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    36.0/9, 54.0/9, 54.0/9, 54.0/9, 36.0/9,
    24.0/9, 36.0/9, 36.0/9, 36.0/9, 24.0/9
};

void TestCppConvnetConvLayer() {
  bool print_outputs = false;
  bool padding_en = false;
  bool bias_en = false;

  int ker_size = 3;
  int group = 2;
  int stride = 1;
  int N = 1;
  int C = 4;
  int H = 5;
  int W = 5;
  int M = 4;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = NULL;
  if (bias_en) {
    bias = malloc(out_dim.c * sizeof(float));
  }
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
  SeqInitF32(in_data, TensorSize(in_dim));
  SeqInitF32(filters, TensorSize(filt_dim));
  if (bias_en) {
    RandInitF32(bias, out_dim.c);
  }

  RefConv2dF32(in_data, filters,
      bias, in_dim.c, in_dim.h,
      in_dim.w, out_dim.c, out_dim.h, out_dim.w,
      ker_size, group,
      pad, stride, bias_en, ref_output);

  CppConvnetConvLayer(in_data, filters, bias,
                           in_dim, filt_dim, stride,
                           pad, group, output);

  if (print_outputs) {
    printf("Output of kn2xyz method\n");
    PrintTensor(output, out_dim);
    printf("Output of reference implementation\n");
    PrintTensor(ref_output, out_dim);
  }
  if (TensorCompare(output, ref_output, out_dim)) {
    printf("PASS\n");
  } else {
    printf("FAIL\n");
  }
  free(in_data);
  free(bias);
  free(ref_output);
  free(filters);
  free(output);
}

void TestLayoutConverters() {
  // NCHW to HWNC converter.
  int N = 2;
  int C = 4;
  int H = 3;
  int W = 3;
  float *nchw_data = malloc(N * C * H * W * sizeof(float));
  float *hwnc_output = malloc(N * C * H * W * sizeof(float));
  RandInitF32(nchw_data, N * C * H * W);

  NCHW2HWNC(nchw_data, N, C, H, W, hwnc_output);
  TensorDim in_dim = {N, C, H, W};
  TensorDim out_dim = {H, W, N, C};
  PrintTensor(nchw_data, in_dim);
  PrintTensor(hwnc_output, out_dim);

  free(nchw_data);
  free(hwnc_output);
}


void TestIm2RowConvLayer() {
  bool padding_en = false;
  bool bias_en = false;

  int ker_size = 3;
  int group = 1;
  int stride = 1;
  int N = 1;
  int C = 64;
  int H = 224;
  int W = 224;
  int M = 64;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = NULL;
  if (bias_en) {
    bias = malloc(out_dim.c * sizeof(float));
  }
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  SeqInitF32(in_data, TensorSize(in_dim));
  SeqInitF32(filters, TensorSize(filt_dim));
  if (bias_en) {
    RandInitF32(bias, out_dim.c);
  }

  gettimeofday(&start, NULL);
  CppConvnetConvLayer(in_data, filters, bias,
                           in_dim, filt_dim, stride,
                           pad, group, output);
  gettimeofday(&end, NULL);
  time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
  printf("Time taken: %lld", time_taken);
  
  free(in_data);
  free(bias);
  free(filters);
  free(output);
}


void TestKer2RowConvLayerKnownOutput() {
  int pad = conv_test_k/2;
  int group = 1;
  int stride = 1;
  TensorDim in_dim = {conv_test_batch, conv_test_in_c,
      conv_test_in_h, conv_test_in_w};
  TensorDim filt_dim = {conv_test_out_c, conv_test_in_c, conv_test_k,
      conv_test_k};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = conv_test_out_c;
  out_dim.n = in_dim.n;
  const float *in_data = conv_test_in_data;
  const float *filters = conv_test_filter;
  const float *bias = conv_test_bias;
  float *output = malloc(out_dim.n * out_dim.c * out_dim.h * out_dim.w *
                         sizeof(float));
  Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                        group, output);

  PrintTensor(output, out_dim);
  TensorCompare(output, conv_test_ref_out, out_dim);
  free(output);
}

void TestKer2FlavorConvLayer(bool is_ker2row, int num_filters, int kernel_size, int batch_size, int depth, int height, int width) {
  // Configurations
  // Enable kn2row or kn2col
  bool kn2row = is_ker2row;
  bool padding_en = true;
  bool bias_en = true;

  int ker_size = kernel_size;
  int group = 1;
  int stride = 1;
  int N = batch_size;
  int C = depth;
  int H = height;
  int W = width;
  int M = num_filters;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = malloc(out_dim.c * sizeof(float));
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  RandInitF32(in_data, TensorSize(in_dim));
  RandInitF32(filters, TensorSize(filt_dim));
  if (bias_en) {
    RandInitF32(bias, out_dim.c);
  }

  gettimeofday(&start, NULL);
  if (kn2row) {
    printf("Using Kn2Row convolution\n");
    
    Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                     group, output);
  } else {
    printf("Using Kn2Col convolution\n");
    Kn2ColConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                         group, output);
  }
  gettimeofday(&end, NULL);
  time_taken = (long long)end.tv_usec - (long long)start.tv_usec;

  printf("Used time: %lld", time_taken);

  free(in_data);
  free(bias);
  free(filters);
  free(output);
}

void TestIm2FlavorConvLayer(bool is_im2col, int num_filters, int kernel_size, int batch_size, int depth, int height, int width) {
  // Configurations
  // Enable kn2row or kn2col
  bool im2col = is_im2col;
  bool padding_en = true;
  bool bias_en = false;

  int ker_size = kernel_size;
  int group = 3;
  int stride = 1;
  int N = batch_size;
  int C = depth;
  int H = height;
  int W = width;
  int M = num_filters;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = malloc(out_dim.c * sizeof(float));
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
  float *spad = malloc(out_dim.h * out_dim.w * in_dim.c *
                        filt_dim.w * filt_dim.h * sizeof(float) );
  RandInitF32(in_data, TensorSize(in_dim));
  RandInitF32(filters, TensorSize(filt_dim));
  if (bias_en) {
    RandInitF32(bias, out_dim.c);
  }

  gettimeofday(&start, NULL);
  if (im2col) {
    printf("Using im2col convolution\n");
    Im2ColConvLayer(in_data, filters, bias, spad, in_dim, out_dim, ker_size,
                    group, pad, stride, bias_en, output);
  } else {

  }
  gettimeofday(&end, NULL);

  time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
  printf("Used time: %lld", time_taken);

  
  free(in_data);
  free(bias);
  free(ref_output);
  free(filters);
  free(output);
  free(spad);
}

void TestMatShiftAdd() {
  int mat_h = 5;
  int mat_w = 5;
  int row_shift = -1;
  int col_shift = -1;
  float *mat1 = malloc(mat_h * mat_w * sizeof(float));
  float *mat2 = malloc(mat_h * mat_w * sizeof(float));
  RandInitF32(mat1, mat_h * mat_w);
  RandInitF32(mat2, mat_h * mat_w);

  PrintMat("base mat", mat1, mat_h, mat_w, CblasRowMajor);
  PrintMat("overlap mat", mat2, mat_h, mat_w, CblasRowMajor);

  MatrixShiftAdd(mat1, mat_h, mat_w, mat2, mat_h, mat_w, row_shift, col_shift);

  PrintMat("result mat", mat1, mat_h, mat_w, CblasRowMajor);
  free(mat1);
  free(mat2);
}

void printArray(double arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%.5f ", arr[i]);
    }
}


void Test_VGG16(int block, int ordinal, int num_iterations, int method, char *save_file) {
  FILE *file = fopen(save_file, "w"); // "w" là chế độ ghi (write)

  bool bias_en = false;

  int ker_size = 3;
  int group = 1;
  int stride = 1;
  int N = 1;
  int C;
  int H;
  int W;
  int M;
  int pad = ker_size / 2;


  if (block == 1) {
    H = 224;
    W = 224;
    if (ordinal == 1) {
      C = 3;
      M = 64;
    }
    if (ordinal == 2) {
      C = 64;
      M = 64;
    }
  }
  
  if (block == 2) {
    H = 112;
    W = 112;
    if (ordinal == 1) {
      C = 64;
      M = 128;
    }
    if (ordinal == 2) {
      C = 128;
      M = 128;
    }
  }

  if (block == 3) {
    H = 56;
    W = 56;
    if (ordinal == 1) {
      C = 128;
      M = 256;
    }
    if (ordinal == 2) {
      C = 256;
      M = 256;
    }
  }

  if (block == 4) {
    H = 28;
    W = 28;
    if (ordinal == 1) {
      C = 256;
      M = 512;
    }
    if (ordinal == 2) {
      C = 512;
      M = 512;
    }
  }

  if (block == 5) {
    H = 14;
    W = 14;
    if (ordinal == 1) {
      C = 512;
      M = 512;
    }
    if (ordinal == 2) {
      C = 512;
      M = 512;
    }
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;

  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;

  if (method == 1) {
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
      RandInitF32(bias, out_dim.c);
    }

    printf("Using Kn2Row convolution\n");
    
    for (int i = 0; i < num_iterations; i++) {
      gettimeofday(&start, NULL);
      Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                      group, output);
      gettimeofday(&end, NULL);
      time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
      fprintf(file, "%lld ", time_taken);
    }

    free(in_data);
    free(bias);
    free(filters);
    free(output);
  }

  if (method == 2) { // using kn2col
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
      RandInitF32(bias, out_dim.c);
    }

    printf("Using Kn2Col convolution\n");
    
    for (int i = 0; i < num_iterations; i++) {
      gettimeofday(&start, NULL);
      Kn2RowConvLayer(in_data, filters, bias, in_dim, filt_dim, stride, pad,
                      group, output);
      gettimeofday(&end, NULL);
      time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
      fprintf(file, "%lld ", time_taken);
    }

    free(in_data);
    free(bias);
    free(filters);
    free(output);
  }
  
  if (method == 3) { // using im2col
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = malloc(out_dim.c * sizeof(float));
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    float *spad = malloc(out_dim.h * out_dim.w * in_dim.c *
                          filt_dim.w * filt_dim.h * sizeof(float) );
    RandInitF32(in_data, TensorSize(in_dim));
    RandInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
      RandInitF32(bias, out_dim.c);
    }

    printf("Using im2col convolution\n");
    for (int i = 0; i < num_iterations; i++) {
      gettimeofday(&start, NULL);
      Im2ColConvLayer(in_data, filters, bias, spad, in_dim, out_dim, ker_size,
                      group, pad, stride, bias_en, output);
      gettimeofday(&end, NULL);
      time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
      fprintf(file, "%lld ", time_taken);
    }

    free(in_data);
    free(filters);
    free(bias);
    free(output);
    free(spad);
  }

  if (method == 4) { // using im2row
    float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
    float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
    float *bias = NULL;
    if (bias_en) {
      bias = malloc(out_dim.c * sizeof(float));
    }
    float *output = malloc(TensorSize(out_dim) * sizeof(float));
    SeqInitF32(in_data, TensorSize(in_dim));
    SeqInitF32(filters, TensorSize(filt_dim));
    if (bias_en) {
      RandInitF32(bias, out_dim.c);
    }

    printf("Using Im2Row Convolution:");
    for (int i = 0; i < num_iterations; i++) {
      gettimeofday(&start, NULL);
      CppConvnetConvLayer(in_data, filters, bias,
                              in_dim, filt_dim, stride,
                              pad, group, output);
      gettimeofday(&end, NULL);
      time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
      fprintf(file, "%lld ", time_taken);
    }

    free(in_data);
    free(bias);
    free(filters);
    free(output);
  }
  
  fclose(file);
}

void TestIm2FlavourWithThread(int num_iterations, int num_threads, int method) {
  bool padding_en = true;

  int ker_size = 3;
  int group = 1;
  int stride = 1;
  int N = 1;
  int C = 3;
  int H = 896;
  int W = 896;
  int M = 128;
  int pad = 0;
  if (padding_en) {
    pad = ker_size / 2;
  }

  TensorDim in_dim = {N, C, H, W};
  TensorDim filt_dim = {M, C, ker_size, ker_size};
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = M;
  out_dim.n = in_dim.n;
  float *in_data = malloc(TensorSize(in_dim) * sizeof(float));
  float *filters = malloc(TensorSize(filt_dim) * sizeof(float));
  float *bias = malloc(out_dim.c * sizeof(float));
  float *output = malloc(TensorSize(out_dim) * sizeof(float));
  float *ref_output = malloc(TensorSize(out_dim) * sizeof(float));
  float *spad = malloc(out_dim.h * out_dim.w * in_dim.c *
                        filt_dim.w * filt_dim.h * sizeof(float) );
  RandInitF32(in_data, TensorSize(in_dim));
  RandInitF32(filters, TensorSize(filt_dim));

  openblas_set_num_threads(num_threads);
  gettimeofday(&start, NULL);
  if (method == 1) {
    printf("Using im2col convolution\n");
    Im2ColConvLayer(in_data, filters, bias, spad, in_dim, out_dim, ker_size,
                    group, pad, stride, false, output);
  } else {
    printf("Using im2col convolution\n");
    CppConvnetConvLayer(in_data, filters, bias,
                              in_dim, filt_dim, stride,
                              pad, group, output);
  }
  gettimeofday(&end, NULL);

  time_taken = (long long)end.tv_usec - (long long)start.tv_usec;
  printf("Used time: %lld", time_taken);

  
  free(in_data);
  free(bias);
  free(ref_output);
  free(filters);
  free(output);
  free(spad);
}

int main(void) {
  // TestLayoutConverters();
  // TestKer2RowConvLayerKnownOutput();
  // TestKer2FlavorConvLayer(false, 64, 3, 1, 3, 224, 224);
  // TestMatShiftAdd();
  //TestCppConvnetConvLayer();

  // TestIm2FlavorConvLayer(true, 64, 3, 1, 3, 224, 224);
  // WinoGradConvHook();

  // TestVGG16_CONV1_2(1, 1);
  // TestVGG16_CONV5_2(1000, 2);
  // TestIm2RowConvLayer();

  // initial run
  // Test_VGG16(1, 2, 1000, 3, "vgg16_conv1_2_im2col.txt");


  // // 4 cores
  // // Conv 1_1
  // Test_VGG16(1, 1, 1000, 1, "./time_execution_4_cores/vgg16_conv1_1_kn2row.txt");
  // Test_VGG16(1, 1, 1000, 2, "./time_execution_4_cores/vgg16_conv1_1_kn2col.txt");
  // Test_VGG16(1, 1, 1000, 3, "./time_execution_4_cores/vgg16_conv1_1_im2col.txt");
  // Test_VGG16(1, 1, 1000, 4, "./time_execution_4_cores/vgg16_conv1_1_im2row.txt");  

  // // Conv 1_2
  // Test_VGG16(1, 2, 1000, 1, "./time_execution_4_cores/vgg16_conv1_2_kn2row.txt");
  // Test_VGG16(1, 2, 1000, 2, "./time_execution_4_cores/vgg16_conv1_2_kn2col.txt");
  // Test_VGG16(1, 2, 1000, 3, "./time_execution_4_cores/vgg16_conv1_2_im2col.txt");
  // Test_VGG16(1, 2, 1000, 4, "./time_execution_4_cores/vgg16_conv1_2_im2row.txt");

  // // Conv 2_1
  // Test_VGG16(2, 1, 1000, 1, "./time_execution_4_cores/vgg16_conv2_1_kn2row.txt");
  // Test_VGG16(2, 1, 1000, 2, "./time_execution_4_cores/vgg16_conv2_1_kn2col.txt");
  // Test_VGG16(2, 1, 1000, 3, "./time_execution_4_cores/vgg16_conv2_1_im2col.txt");
  // Test_VGG16(2, 1, 1000, 4, "./time_execution_4_cores/vgg16_conv2_1_im2row.txt");

  // // Conv 2_2
  // Test_VGG16(2, 2, 1000, 1, "./time_execution_4_cores/vgg16_conv2_2_kn2row.txt");
  // Test_VGG16(2, 2, 1000, 2, "./time_execution_4_cores/vgg16_conv2_2_kn2col.txt");
  // Test_VGG16(2, 2, 1000, 3, "./time_execution_4_cores/vgg16_conv2_2_im2col.txt");
  // Test_VGG16(2, 2, 1000, 4, "./time_execution_4_cores/vgg16_conv2_2_im2row.txt");

  // // Conv 3_1
  // Test_VGG16(3, 1, 1000, 1, "./time_execution_4_cores/vgg16_conv3_1_kn2row.txt");
  // Test_VGG16(3, 1, 1000, 2, "./time_execution_4_cores/vgg16_conv3_1_kn2col.txt");
  // Test_VGG16(3, 1, 1000, 3, "./time_execution_4_cores/vgg16_conv3_1_im2col.txt");
  // Test_VGG16(3, 1, 1000, 4, "./time_execution_4_cores/vgg16_conv3_1_im2row.txt");

  // // COnv 3_2
  // Test_VGG16(3, 2, 1000, 1, "./time_execution_4_cores/vgg16_conv3_2_kn2row.txt");
  // Test_VGG16(3, 2, 1000, 2, "./time_execution_4_cores/vgg16_conv3_2_kn2col.txt");
  // Test_VGG16(3, 2, 1000, 3, "./time_execution_4_cores/vgg16_conv3_2_im2col.txt");
  // Test_VGG16(3, 2, 1000, 4, "./time_execution_4_cores/vgg16_conv3_2_im2row.txt");

  // Conv 4_1
  // Test_VGG16(4, 1, 1000, 1, "./time_execution_4_cores/vgg16_conv4_1_kn2row.txt");
  // Test_VGG16(4, 1, 1000, 2, "./time_execution_4_cores/vgg16_conv4_1_kn2col.txt");
  // Test_VGG16(4, 1, 1000, 3, "./time_execution_4_cores/vgg16_conv4_1_im2col.txt");
  // Test_VGG16(4, 1, 1000, 4, "./time_execution_4_cores/vgg16_conv4_1_im2row.txt");

  // // Conv 4_2
  // Test_VGG16(4, 2, 1000, 1, "./time_execution_4_cores/vgg16_conv4_2_kn2row.txt");
  // Test_VGG16(4, 2, 1000, 2, "./time_execution_4_cores/vgg16_conv4_2_kn2col.txt");
  // Test_VGG16(4, 2, 1000, 3, "./time_execution_4_cores/vgg16_conv4_2_im2col.txt");
  // Test_VGG16(4, 2, 1000, 4, "./time_execution_4_cores/vgg16_conv4_2_im2row.txt");

  // // Conv 5_1
  // Test_VGG16(5, 1, 1000, 1, "./time_execution_4_cores/vgg16_conv5_1_kn2row.txt");
  // Test_VGG16(5, 1, 1000, 2, "./time_execution_4_cores/vgg16_conv5_1_kn2col.txt");
  // Test_VGG16(5, 1, 1000, 3, "./time_execution_4_cores/vgg16_conv5_1_im2col.txt");
  // Test_VGG16(5, 1, 1000, 4, "./time_execution_4_cores/vgg16_conv5_1_im2row.txt");

  // // Conv 5_2
  // Test_VGG16(5, 2, 1000, 1, "./time_execution_4_cores/vgg16_conv5_2_kn2row.txt");
  // Test_VGG16(5, 2, 1000, 2, "./time_execution_4_cores/vgg16_conv5_2_kn2col.txt");
  // Test_VGG16(5, 2, 1000, 3, "./time_execution_4_cores/vgg16_conv5_2_im2col.txt");
  // Test_VGG16(5, 2, 1000, 4, "./time_execution_4_cores/vgg16_conv5_2_im2row.txt");

  // // 32 cores
  // // Conv 1_1
  // Test_VGG16(1, 1, 1000, 1, "./time_execution_32_cores/vgg16_conv1_1_kn2row.txt");
  // Test_VGG16(1, 1, 1000, 2, "./time_execution_32_cores/vgg16_conv1_1_kn2col.txt");
  // Test_VGG16(1, 1, 1000, 3, "./time_execution_32_cores/vgg16_conv1_1_im2col.txt");
  // Test_VGG16(1, 1, 1000, 4, "./time_execution_32_cores/vgg16_conv1_1_im2row.txt");  

  // // Conv 1_2
  // Test_VGG16(1, 2, 1000, 1, "./time_execution_32_cores/vgg16_conv1_2_kn2row.txt");
  // Test_VGG16(1, 2, 1000, 2, "./time_execution_32_cores/vgg16_conv1_2_kn2col.txt");
  // Test_VGG16(1, 2, 1000, 3, "./time_execution_32_cores/vgg16_conv1_2_im2col.txt");
  // Test_VGG16(1, 2, 1000, 4, "./time_execution_32_cores/vgg16_conv1_2_im2row.txt");

  // // Conv 2_1
  // Test_VGG16(2, 1, 1000, 1, "./time_execution_32_cores/vgg16_conv2_1_kn2row.txt");
  // Test_VGG16(2, 1, 1000, 2, "./time_execution_32_cores/vgg16_conv2_1_kn2col.txt");
  // Test_VGG16(2, 1, 1000, 3, "./time_execution_32_cores/vgg16_conv2_1_im2col.txt");
  // Test_VGG16(2, 1, 1000, 4, "./time_execution_32_cores/vgg16_conv2_1_im2row.txt");

  // // Conv 2_2
  // Test_VGG16(2, 2, 1000, 1, "./time_execution_32_cores/vgg16_conv2_2_kn2row.txt");
  // Test_VGG16(2, 2, 1000, 2, "./time_execution_32_cores/vgg16_conv2_2_kn2col.txt");
  // Test_VGG16(2, 2, 1000, 3, "./time_execution_32_cores/vgg16_conv2_2_im2col.txt");
  // Test_VGG16(2, 2, 1000, 4, "./time_execution_32_cores/vgg16_conv2_2_im2row.txt");

  // // Conv 3_1
  // Test_VGG16(3, 1, 1000, 1, "./time_execution_32_cores/vgg16_conv3_1_kn2row.txt");
  // Test_VGG16(3, 1, 1000, 2, "./time_execution_32_cores/vgg16_conv3_1_kn2col.txt");
  // Test_VGG16(3, 1, 1000, 3, "./time_execution_32_cores/vgg16_conv3_1_im2col.txt");
  // Test_VGG16(3, 1, 1000, 4, "./time_execution_32_cores/vgg16_conv3_1_im2row.txt");

  // // Conv 3_2
  // Test_VGG16(3, 2, 1000, 1, "./time_execution_32_cores/vgg16_conv3_2_kn2row.txt");
  // Test_VGG16(3, 2, 1000, 2, "./time_execution_32_cores/vgg16_conv3_2_kn2col.txt");
  // Test_VGG16(3, 2, 1000, 3, "./time_execution_32_cores/vgg16_conv3_2_im2col.txt");
  // Test_VGG16(3, 2, 1000, 4, "./time_execution_32_cores/vgg16_conv3_2_im2row.txt");

  // //Conv 4_1
  // Test_VGG16(4, 1, 1000, 1, "./time_execution_32_cores/vgg16_conv4_1_kn2row.txt");
  // Test_VGG16(4, 1, 1000, 2, "./time_execution_32_cores/vgg16_conv4_1_kn2col.txt");
  // Test_VGG16(4, 1, 1000, 3, "./time_execution_32_cores/vgg16_conv4_1_im2col.txt");
  // Test_VGG16(4, 1, 1000, 4, "./time_execution_32_cores/vgg16_conv4_1_im2row.txt");

  // // Conv 4_2
  // Test_VGG16(4, 2, 1000, 1, "./time_execution_32_cores/vgg16_conv4_2_kn2row.txt");
  // Test_VGG16(4, 2, 1000, 2, "./time_execution_32_cores/vgg16_conv4_2_kn2col.txt");
  // Test_VGG16(4, 2, 1000, 3, "./time_execution_32_cores/vgg16_conv4_2_im2col.txt");
  // Test_VGG16(4, 2, 1000, 4, "./time_execution_32_cores/vgg16_conv4_2_im2row.txt");

  // // Conv 5_1
  // Test_VGG16(5, 1, 1000, 1, "./time_execution_32_cores/vgg16_conv5_1_kn2row.txt");
  // Test_VGG16(5, 1, 1000, 2, "./time_execution_32_cores/vgg16_conv5_1_kn2col.txt");
  // Test_VGG16(5, 1, 1000, 3, "./time_execution_32_cores/vgg16_conv5_1_im2col.txt");
  // Test_VGG16(5, 1, 1000, 4, "./time_execution_32_cores/vgg16_conv5_1_im2row.txt");

  // // Conv 5_2
  // Test_VGG16(5, 2, 1000, 1, "./time_execution_32_cores/vgg16_conv5_2_kn2row.txt");
  // Test_VGG16(5, 2, 1000, 2, "./time_execution_32_cores/vgg16_conv5_2_kn2col.txt");
  // Test_VGG16(5, 2, 1000, 3, "./time_execution_32_cores/vgg16_conv5_2_im2col.txt");
  // Test_VGG16(5, 2, 1000, 4, "./time_execution_32_cores/vgg16_conv5_2_im2row.txt");

// // 16 cores
// openblas_set_num_threads(16);
//   // Conv 1_1
//   Test_VGG16(1, 1, 1000, 1, "./time_execution_16_cores/vgg16_conv1_1_kn2row.txt");
//   Test_VGG16(1, 1, 1000, 2, "./time_execution_16_cores/vgg16_conv1_1_kn2col.txt");
//   Test_VGG16(1, 1, 1000, 3, "./time_execution_16_cores/vgg16_conv1_1_im2col.txt");
//   Test_VGG16(1, 1, 1000, 4, "./time_execution_16_cores/vgg16_conv1_1_im2row.txt");  

//   // Conv 1_2
//   Test_VGG16(1, 2, 1000, 1, "./time_execution_16_cores/vgg16_conv1_2_kn2row.txt");
//   Test_VGG16(1, 2, 1000, 2, "./time_execution_16_cores/vgg16_conv1_2_kn2col.txt");
//   Test_VGG16(1, 2, 1000, 3, "./time_execution_16_cores/vgg16_conv1_2_im2col.txt");
//   Test_VGG16(1, 2, 1000, 4, "./time_execution_16_cores/vgg16_conv1_2_im2row.txt");

//   // Conv 2_1
//   Test_VGG16(2, 1, 1000, 1, "./time_execution_16_cores/vgg16_conv2_1_kn2row.txt");
//   Test_VGG16(2, 1, 1000, 2, "./time_execution_16_cores/vgg16_conv2_1_kn2col.txt");
//   Test_VGG16(2, 1, 1000, 3, "./time_execution_16_cores/vgg16_conv2_1_im2col.txt");
//   Test_VGG16(2, 1, 1000, 4, "./time_execution_16_cores/vgg16_conv2_1_im2row.txt");

//   // Conv 2_2
//   Test_VGG16(2, 2, 1000, 1, "./time_execution_16_cores/vgg16_conv2_2_kn2row.txt");
//   Test_VGG16(2, 2, 1000, 2, "./time_execution_16_cores/vgg16_conv2_2_kn2col.txt");
//   Test_VGG16(2, 2, 1000, 3, "./time_execution_16_cores/vgg16_conv2_2_im2col.txt");
//   Test_VGG16(2, 2, 1000, 4, "./time_execution_16_cores/vgg16_conv2_2_im2row.txt");

//   // Conv 3_1
//   Test_VGG16(3, 1, 1000, 1, "./time_execution_16_cores/vgg16_conv3_1_kn2row.txt");
//   Test_VGG16(3, 1, 1000, 2, "./time_execution_16_cores/vgg16_conv3_1_kn2col.txt");
//   Test_VGG16(3, 1, 1000, 3, "./time_execution_16_cores/vgg16_conv3_1_im2col.txt");
//   Test_VGG16(3, 1, 1000, 4, "./time_execution_16_cores/vgg16_conv3_1_im2row.txt");

//   // Conv 3_2
//   Test_VGG16(3, 2, 1000, 1, "./time_execution_16_cores/vgg16_conv3_2_kn2row.txt");
//   Test_VGG16(3, 2, 1000, 2, "./time_execution_16_cores/vgg16_conv3_2_kn2col.txt");
//   Test_VGG16(3, 2, 1000, 3, "./time_execution_16_cores/vgg16_conv3_2_im2col.txt");
//   Test_VGG16(3, 2, 1000, 4, "./time_execution_16_cores/vgg16_conv3_2_im2row.txt");

//   //Conv 4_1
//   Test_VGG16(4, 1, 1000, 1, "./time_execution_16_cores/vgg16_conv4_1_kn2row.txt");
//   Test_VGG16(4, 1, 1000, 2, "./time_execution_16_cores/vgg16_conv4_1_kn2col.txt");
//   Test_VGG16(4, 1, 1000, 3, "./time_execution_16_cores/vgg16_conv4_1_im2col.txt");
//   Test_VGG16(4, 1, 1000, 4, "./time_execution_16_cores/vgg16_conv4_1_im2row.txt");

//   // Conv 4_2
//   Test_VGG16(4, 2, 1000, 1, "./time_execution_16_cores/vgg16_conv4_2_kn2row.txt");
//   Test_VGG16(4, 2, 1000, 2, "./time_execution_16_cores/vgg16_conv4_2_kn2col.txt");
//   Test_VGG16(4, 2, 1000, 3, "./time_execution_16_cores/vgg16_conv4_2_im2col.txt");
//   Test_VGG16(4, 2, 1000, 4, "./time_execution_16_cores/vgg16_conv4_2_im2row.txt");

//   // Conv 5_1
//   Test_VGG16(5, 1, 1000, 1, "./time_execution_16_cores/vgg16_conv5_1_kn2row.txt");
//   Test_VGG16(5, 1, 1000, 2, "./time_execution_16_cores/vgg16_conv5_1_kn2col.txt");
//   Test_VGG16(5, 1, 1000, 3, "./time_execution_16_cores/vgg16_conv5_1_im2col.txt");
//   Test_VGG16(5, 1, 1000, 4, "./time_execution_16_cores/vgg16_conv5_1_im2row.txt");

//   // Conv 5_2
//   Test_VGG16(5, 2, 1000, 1, "./time_execution_16_cores/vgg16_conv5_2_kn2row.txt");
//   Test_VGG16(5, 2, 1000, 2, "./time_execution_16_cores/vgg16_conv5_2_kn2col.txt");
//   Test_VGG16(5, 2, 1000, 3, "./time_execution_16_cores/vgg16_conv5_2_im2col.txt");
//   Test_VGG16(5, 2, 1000, 4, "./time_execution_16_cores/vgg16_conv5_2_im2row.txt");

// // 8 cores
// openblas_set_num_threads(8);
//   // Conv 1_1
//   Test_VGG16(1, 1, 1000, 1, "./time_execution_8_cores/vgg16_conv1_1_kn2row.txt");
//   Test_VGG16(1, 1, 1000, 2, "./time_execution_8_cores/vgg16_conv1_1_kn2col.txt");
//   Test_VGG16(1, 1, 1000, 3, "./time_execution_8_cores/vgg16_conv1_1_im2col.txt");
//   Test_VGG16(1, 1, 1000, 4, "./time_execution_8_cores/vgg16_conv1_1_im2row.txt");  

//   // Conv 1_2
//   Test_VGG16(1, 2, 1000, 1, "./time_execution_8_cores/vgg16_conv1_2_kn2row.txt");
//   Test_VGG16(1, 2, 1000, 2, "./time_execution_8_cores/vgg16_conv1_2_kn2col.txt");
//   Test_VGG16(1, 2, 1000, 3, "./time_execution_8_cores/vgg16_conv1_2_im2col.txt");
//   Test_VGG16(1, 2, 1000, 4, "./time_execution_8_cores/vgg16_conv1_2_im2row.txt");

//   // Conv 2_1
//   Test_VGG16(2, 1, 1000, 1, "./time_execution_8_cores/vgg16_conv2_1_kn2row.txt");
//   Test_VGG16(2, 1, 1000, 2, "./time_execution_8_cores/vgg16_conv2_1_kn2col.txt");
//   Test_VGG16(2, 1, 1000, 3, "./time_execution_8_cores/vgg16_conv2_1_im2col.txt");
//   Test_VGG16(2, 1, 1000, 4, "./time_execution_8_cores/vgg16_conv2_1_im2row.txt");

//   // Conv 2_2
//   Test_VGG16(2, 2, 1000, 1, "./time_execution_8_cores/vgg16_conv2_2_kn2row.txt");
//   Test_VGG16(2, 2, 1000, 2, "./time_execution_8_cores/vgg16_conv2_2_kn2col.txt");
//   Test_VGG16(2, 2, 1000, 3, "./time_execution_8_cores/vgg16_conv2_2_im2col.txt");
//   Test_VGG16(2, 2, 1000, 4, "./time_execution_8_cores/vgg16_conv2_2_im2row.txt");

//   // Conv 3_1
//   Test_VGG16(3, 1, 1000, 1, "./time_execution_8_cores/vgg16_conv3_1_kn2row.txt");
//   Test_VGG16(3, 1, 1000, 2, "./time_execution_8_cores/vgg16_conv3_1_kn2col.txt");
//   Test_VGG16(3, 1, 1000, 3, "./time_execution_8_cores/vgg16_conv3_1_im2col.txt");
//   Test_VGG16(3, 1, 1000, 4, "./time_execution_8_cores/vgg16_conv3_1_im2row.txt");

//   // Conv 3_2
//   Test_VGG16(3, 2, 1000, 1, "./time_execution_8_cores/vgg16_conv3_2_kn2row.txt");
//   Test_VGG16(3, 2, 1000, 2, "./time_execution_8_cores/vgg16_conv3_2_kn2col.txt");
//   Test_VGG16(3, 2, 1000, 3, "./time_execution_8_cores/vgg16_conv3_2_im2col.txt");
//   Test_VGG16(3, 2, 1000, 4, "./time_execution_8_cores/vgg16_conv3_2_im2row.txt");

//   //Conv 4_1
//   Test_VGG16(4, 1, 1000, 1, "./time_execution_8_cores/vgg16_conv4_1_kn2row.txt");
//   Test_VGG16(4, 1, 1000, 2, "./time_execution_8_cores/vgg16_conv4_1_kn2col.txt");
//   Test_VGG16(4, 1, 1000, 3, "./time_execution_8_cores/vgg16_conv4_1_im2col.txt");
//   Test_VGG16(4, 1, 1000, 4, "./time_execution_8_cores/vgg16_conv4_1_im2row.txt");

//   // Conv 4_2
//   Test_VGG16(4, 2, 1000, 1, "./time_execution_8_cores/vgg16_conv4_2_kn2row.txt");
//   Test_VGG16(4, 2, 1000, 2, "./time_execution_8_cores/vgg16_conv4_2_kn2col.txt");
//   Test_VGG16(4, 2, 1000, 3, "./time_execution_8_cores/vgg16_conv4_2_im2col.txt");
//   Test_VGG16(4, 2, 1000, 4, "./time_execution_8_cores/vgg16_conv4_2_im2row.txt");

//   // Conv 5_1
//   Test_VGG16(5, 1, 1000, 1, "./time_execution_8_cores/vgg16_conv5_1_kn2row.txt");
//   Test_VGG16(5, 1, 1000, 2, "./time_execution_8_cores/vgg16_conv5_1_kn2col.txt");
//   Test_VGG16(5, 1, 1000, 3, "./time_execution_8_cores/vgg16_conv5_1_im2col.txt");
//   Test_VGG16(5, 1, 1000, 4, "./time_execution_8_cores/vgg16_conv5_1_im2row.txt");

//   // Conv 5_2
//   Test_VGG16(5, 2, 1000, 1, "./time_execution_8_cores/vgg16_conv5_2_kn2row.txt");
//   Test_VGG16(5, 2, 1000, 2, "./time_execution_8_cores/vgg16_conv5_2_kn2col.txt");
//   Test_VGG16(5, 2, 1000, 3, "./time_execution_8_cores/vgg16_conv5_2_im2col.txt");
//   Test_VGG16(5, 2, 1000, 4, "./time_execution_8_cores/vgg16_conv5_2_im2row.txt");

// // 32 cores with set
// openblas_set_num_threads(32);
//   // Conv 1_1
//   Test_VGG16(1, 1, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv1_1_kn2row.txt");
//   Test_VGG16(1, 1, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv1_1_kn2col.txt");
//   Test_VGG16(1, 1, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv1_1_im2col.txt");
//   Test_VGG16(1, 1, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv1_1_im2row.txt");  

//   // Conv 1_2
//   Test_VGG16(1, 2, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv1_2_kn2row.txt");
//   Test_VGG16(1, 2, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv1_2_kn2col.txt");
//   Test_VGG16(1, 2, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv1_2_im2col.txt");
//   Test_VGG16(1, 2, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv1_2_im2row.txt");

//   // Conv 2_1
//   Test_VGG16(2, 1, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv2_1_kn2row.txt");
//   Test_VGG16(2, 1, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv2_1_kn2col.txt");
//   Test_VGG16(2, 1, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv2_1_im2col.txt");
//   Test_VGG16(2, 1, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv2_1_im2row.txt");

//   // Conv 2_2
//   Test_VGG16(2, 2, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv2_2_kn2row.txt");
//   Test_VGG16(2, 2, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv2_2_kn2col.txt");
//   Test_VGG16(2, 2, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv2_2_im2col.txt");
//   Test_VGG16(2, 2, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv2_2_im2row.txt");

//   // Conv 3_1
//   Test_VGG16(3, 1, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv3_1_kn2row.txt");
//   Test_VGG16(3, 1, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv3_1_kn2col.txt");
//   Test_VGG16(3, 1, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv3_1_im2col.txt");
//   Test_VGG16(3, 1, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv3_1_im2row.txt");

//   // Conv 3_2
//   Test_VGG16(3, 2, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv3_2_kn2row.txt");
//   Test_VGG16(3, 2, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv3_2_kn2col.txt");
//   Test_VGG16(3, 2, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv3_2_im2col.txt");
//   Test_VGG16(3, 2, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv3_2_im2row.txt");

//   //Conv 4_1
//   Test_VGG16(4, 1, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv4_1_kn2row.txt");
//   Test_VGG16(4, 1, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv4_1_kn2col.txt");
//   Test_VGG16(4, 1, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv4_1_im2col.txt");
//   Test_VGG16(4, 1, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv4_1_im2row.txt");

//   // Conv 4_2
//   Test_VGG16(4, 2, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv4_2_kn2row.txt");
//   Test_VGG16(4, 2, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv4_2_kn2col.txt");
//   Test_VGG16(4, 2, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv4_2_im2col.txt");
//   Test_VGG16(4, 2, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv4_2_im2row.txt");

//   // Conv 5_1
//   Test_VGG16(5, 1, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv5_1_kn2row.txt");
//   Test_VGG16(5, 1, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv5_1_kn2col.txt");
//   Test_VGG16(5, 1, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv5_1_im2col.txt");
//   Test_VGG16(5, 1, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv5_1_im2row.txt");

//   // Conv 5_2
//   Test_VGG16(5, 2, 1000, 1, "./time_execution_32_cores_with_set/vgg16_conv5_2_kn2row.txt");
//   Test_VGG16(5, 2, 1000, 2, "./time_execution_32_cores_with_set/vgg16_conv5_2_kn2col.txt");
//   Test_VGG16(5, 2, 1000, 3, "./time_execution_32_cores_with_set/vgg16_conv5_2_im2col.txt");
//   Test_VGG16(5, 2, 1000, 4, "./time_execution_32_cores_with_set/vgg16_conv5_2_im2row.txt");

TestIm2FlavourWithThread(1, 16, 1);
  
  return 0;
}
