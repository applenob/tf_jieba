#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// See tensorflow/core/ops/audio_ops.cc
namespace delta {

using namespace tensorflow;  // NOLINT
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::UnchangedShape;


REGISTER_OP("JiebaCut")
    .Input("sentence_in: string")
    .Output("sentence_out: string")
    .Attr("use_file: bool = false")
    .Attr("dict_lines: list(string) = ['']")
    .Attr("model_lines: list(string) = ['']")
    .Attr("user_dict_lines: list(string) = ['']")
    .Attr("idf_lines: list(string) = ['']")
    .Attr("stop_word_lines: list(string) = ['']")
    .Attr("dict_path: string = ''")
    .Attr("hmm_path: string = ''")
    .Attr("user_dict_path: string = ''")
    .Attr("idf_path: string = ''")
    .Attr("stop_word_path: string = ''")
    .Attr("hmm: bool = true")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Cut the Chines sentence into words.
sentence_in: A scalar or list of strings.
sentence_out: A scalar or list of strings.
)doc");


}  // namespace delta
