; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8224
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 12336
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16448
  br label %7

7:                                                ; preds = %3, %15
  %8 = phi i64 [ 0, %3 ], [ %9, %15 ]
  %9 = add nuw nsw i64 %8, 1
  %10 = add nuw nsw i64 %8, 2
  %11 = add nuw nsw i64 %8, 3
  %12 = add nuw nsw i64 %8, 4
  %13 = getelementptr inbounds nuw float, ptr %2, i64 %8
  br label %17

14:                                               ; preds = %15
  ret void

15:                                               ; preds = %17
  %16 = icmp eq i64 %9, 1024
  br i1 %16, label %14, label %7

17:                                               ; preds = %7, %17
  %18 = phi i64 [ 0, %7 ], [ %36, %17 ]
  %19 = mul nuw nsw i64 %18, 1028
  %20 = getelementptr inbounds nuw float, ptr %0, i64 %19
  %21 = getelementptr inbounds nuw float, ptr %20, i64 %8
  %22 = load float, ptr %21, align 4, !tbaa !10
  %23 = getelementptr inbounds nuw float, ptr %20, i64 %9
  %24 = load float, ptr %23, align 4, !tbaa !10
  %25 = fmul float %24, 4.000000e+00
  %26 = tail call float @llvm.fmuladd.f32(float %22, float 2.000000e+00, float %25)
  %27 = getelementptr inbounds nuw float, ptr %20, i64 %10
  %28 = load float, ptr %27, align 4, !tbaa !10
  %29 = tail call float @llvm.fmuladd.f32(float %28, float 5.000000e+00, float %26)
  %30 = getelementptr inbounds nuw float, ptr %20, i64 %11
  %31 = load float, ptr %30, align 4, !tbaa !10
  %32 = tail call float @llvm.fmuladd.f32(float %31, float 4.000000e+00, float %29)
  %33 = getelementptr inbounds nuw float, ptr %20, i64 %12
  %34 = load float, ptr %33, align 4, !tbaa !10
  %35 = tail call float @llvm.fmuladd.f32(float %34, float 2.000000e+00, float %32)
  %36 = add nuw nsw i64 %18, 1
  %37 = mul nuw nsw i64 %36, 4112
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 %37
  %39 = getelementptr inbounds nuw float, ptr %38, i64 %8
  %40 = load float, ptr %39, align 4, !tbaa !10
  %41 = tail call float @llvm.fmuladd.f32(float %40, float 4.000000e+00, float %35)
  %42 = getelementptr inbounds nuw float, ptr %38, i64 %9
  %43 = load float, ptr %42, align 4, !tbaa !10
  %44 = tail call float @llvm.fmuladd.f32(float %43, float 9.000000e+00, float %41)
  %45 = getelementptr inbounds nuw float, ptr %38, i64 %10
  %46 = load float, ptr %45, align 4, !tbaa !10
  %47 = tail call float @llvm.fmuladd.f32(float %46, float 1.200000e+01, float %44)
  %48 = getelementptr inbounds nuw float, ptr %38, i64 %11
  %49 = load float, ptr %48, align 4, !tbaa !10
  %50 = tail call float @llvm.fmuladd.f32(float %49, float 9.000000e+00, float %47)
  %51 = getelementptr inbounds nuw float, ptr %38, i64 %12
  %52 = load float, ptr %51, align 4, !tbaa !10
  %53 = tail call float @llvm.fmuladd.f32(float %52, float 4.000000e+00, float %50)
  %54 = getelementptr inbounds nuw float, ptr %4, i64 %19
  %55 = getelementptr inbounds nuw float, ptr %54, i64 %8
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = tail call float @llvm.fmuladd.f32(float %56, float 5.000000e+00, float %53)
  %58 = getelementptr inbounds nuw float, ptr %54, i64 %9
  %59 = load float, ptr %58, align 4, !tbaa !10
  %60 = tail call float @llvm.fmuladd.f32(float %59, float 1.200000e+01, float %57)
  %61 = getelementptr inbounds nuw float, ptr %54, i64 %10
  %62 = load float, ptr %61, align 4, !tbaa !10
  %63 = tail call float @llvm.fmuladd.f32(float %62, float 1.500000e+01, float %60)
  %64 = getelementptr inbounds nuw float, ptr %54, i64 %11
  %65 = load float, ptr %64, align 4, !tbaa !10
  %66 = tail call float @llvm.fmuladd.f32(float %65, float 1.200000e+01, float %63)
  %67 = getelementptr inbounds nuw float, ptr %54, i64 %12
  %68 = load float, ptr %67, align 4, !tbaa !10
  %69 = tail call float @llvm.fmuladd.f32(float %68, float 5.000000e+00, float %66)
  %70 = getelementptr inbounds nuw float, ptr %5, i64 %19
  %71 = getelementptr inbounds nuw float, ptr %70, i64 %8
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 4.000000e+00, float %69)
  %74 = getelementptr inbounds nuw float, ptr %70, i64 %9
  %75 = load float, ptr %74, align 4, !tbaa !10
  %76 = tail call float @llvm.fmuladd.f32(float %75, float 9.000000e+00, float %73)
  %77 = getelementptr inbounds nuw float, ptr %70, i64 %10
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = tail call float @llvm.fmuladd.f32(float %78, float 1.200000e+01, float %76)
  %80 = getelementptr inbounds nuw float, ptr %70, i64 %11
  %81 = load float, ptr %80, align 4, !tbaa !10
  %82 = tail call float @llvm.fmuladd.f32(float %81, float 9.000000e+00, float %79)
  %83 = getelementptr inbounds nuw float, ptr %70, i64 %12
  %84 = load float, ptr %83, align 4, !tbaa !10
  %85 = tail call float @llvm.fmuladd.f32(float %84, float 4.000000e+00, float %82)
  %86 = getelementptr inbounds nuw float, ptr %6, i64 %19
  %87 = getelementptr inbounds nuw float, ptr %86, i64 %8
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %88, float 2.000000e+00, float %85)
  %90 = getelementptr inbounds nuw float, ptr %86, i64 %9
  %91 = load float, ptr %90, align 4, !tbaa !10
  %92 = tail call float @llvm.fmuladd.f32(float %91, float 4.000000e+00, float %89)
  %93 = getelementptr inbounds nuw float, ptr %86, i64 %10
  %94 = load float, ptr %93, align 4, !tbaa !10
  %95 = tail call float @llvm.fmuladd.f32(float %94, float 5.000000e+00, float %92)
  %96 = getelementptr inbounds nuw float, ptr %86, i64 %11
  %97 = load float, ptr %96, align 4, !tbaa !10
  %98 = tail call float @llvm.fmuladd.f32(float %97, float 4.000000e+00, float %95)
  %99 = getelementptr inbounds nuw float, ptr %86, i64 %12
  %100 = load float, ptr %99, align 4, !tbaa !10
  %101 = tail call float @llvm.fmuladd.f32(float %100, float 2.000000e+00, float %98)
  %102 = shl nsw i64 %18, 12
  %103 = getelementptr inbounds nuw i8, ptr %13, i64 %102
  store float %101, ptr %103, align 4, !tbaa !10
  %104 = icmp eq i64 %36, 1024
  br i1 %104, label %15, label %17
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"clang version 20.1.0"}
!6 = !{i32 1, i32 1, i32 1}
!7 = !{!"none", !"none", !"none"}
!8 = !{!"float*", !"float*", !"float*"}
!9 = !{!"restrict const", !"restrict", !"restrict"}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
