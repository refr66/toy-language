module {
  func.func private @__sbpf_load_8(i64) -> i8
  func.func private @__sbpf_load_16(i64) -> i16
  func.func private @__sbpf_load_32(i64) -> i32
  func.func private @__sbpf_load_64(i64) -> i64
  func.func private @__sbpf_store_8(i64, i8)
  func.func private @__sbpf_store_16(i64, i16)
  func.func private @__sbpf_store_32(i64, i32)
  func.func private @__sbpf_store_64(i64, i64)
  func.func private @__sbpf_abort()
  func.func @_ZN13relative_call12function_sum17h8da3f9048852644bE(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 {
    %0 = arith.addi %arg1, %arg0 : i64
    return %0 : i64
  }
  func.func @_ZN13relative_call18function_stack_ref17h992bbe44bbc125c9E(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 {
    %c1_i64 = arith.constant 1 : i64
    %0 = call @__sbpf_load_64(%arg0) : (i64) -> i64
    %1 = arith.addi %0, %c1_i64 : i64
    call @__sbpf_store_64(%arg0, %1) : (i64, i64) -> ()
    return %1 : i64
  }
  func.func @entrypoint(%arg0: i64) -> i64 {
    %c8589934336_i64 = arith.constant 8589934336 : i64
    %0 = ub.poison : i64
    %1 = call @__sbpf_load_8(%arg0) : (i64) -> i8
    %2 = arith.extui %1 : i8 to i64
    call @__sbpf_store_64(%c8589934336_i64, %2) : (i64, i64) -> ()
    %3 = call @_ZN13relative_call18function_stack_ref17h992bbe44bbc125c9E(%c8589934336_i64, %0, %0, %0, %0) : (i64, i64, i64, i64, i64) -> i64
    %4 = call @_ZN13relative_call12function_sum17h8da3f9048852644bE(%3, %2, %0, %0, %0) : (i64, i64, i64, i64, i64) -> i64
    return %4 : i64
  }
}

