; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4112
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8224
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 12336
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16448
  br label %8

8:                                                ; preds = %3, %22
  %9 = phi i64 [ 0, %3 ], [ %23, %22 ]
  %10 = shl nuw nsw i64 %9, 6
  br label %12

11:                                               ; preds = %22
  ret void

12:                                               ; preds = %8, %25
  %13 = phi i64 [ 0, %8 ], [ %26, %25 ]
  %14 = mul nuw nsw i64 %13, 1028
  %15 = getelementptr inbounds nuw float, ptr %0, i64 %14
  %16 = getelementptr inbounds nuw float, ptr %4, i64 %14
  %17 = getelementptr inbounds nuw float, ptr %5, i64 %14
  %18 = getelementptr inbounds nuw float, ptr %6, i64 %14
  %19 = getelementptr inbounds nuw float, ptr %7, i64 %14
  %20 = shl nsw i64 %13, 12
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 %20
  br label %28

22:                                               ; preds = %25
  %23 = add nuw nsw i64 %9, 1
  %24 = icmp eq i64 %23, 16
  br i1 %24, label %11, label %8

25:                                               ; preds = %28
  %26 = add nuw nsw i64 %13, 1
  %27 = icmp eq i64 %26, 1024
  br i1 %27, label %22, label %12

28:                                               ; preds = %12, %28
  %29 = phi i64 [ 0, %12 ], [ %111, %28 ]
  %30 = add nuw nsw i64 %29, %10
  %31 = getelementptr inbounds nuw float, ptr %15, i64 %30
  %32 = load float, ptr %31, align 4, !tbaa !10
  %33 = add nuw nsw i64 %30, 1
  %34 = getelementptr inbounds nuw float, ptr %15, i64 %33
  %35 = load float, ptr %34, align 4, !tbaa !10
  %36 = fmul float %35, 4.000000e+00
  %37 = tail call float @llvm.fmuladd.f32(float %32, float 2.000000e+00, float %36)
  %38 = add nuw nsw i64 %30, 2
  %39 = getelementptr inbounds nuw float, ptr %15, i64 %38
  %40 = load float, ptr %39, align 4, !tbaa !10
  %41 = tail call float @llvm.fmuladd.f32(float %40, float 5.000000e+00, float %37)
  %42 = add nuw nsw i64 %30, 3
  %43 = getelementptr inbounds nuw float, ptr %15, i64 %42
  %44 = load float, ptr %43, align 4, !tbaa !10
  %45 = tail call float @llvm.fmuladd.f32(float %44, float 4.000000e+00, float %41)
  %46 = add nuw nsw i64 %30, 4
  %47 = getelementptr inbounds nuw float, ptr %15, i64 %46
  %48 = load float, ptr %47, align 4, !tbaa !10
  %49 = tail call float @llvm.fmuladd.f32(float %48, float 2.000000e+00, float %45)
  %50 = getelementptr inbounds nuw float, ptr %16, i64 %30
  %51 = load float, ptr %50, align 4, !tbaa !10
  %52 = tail call float @llvm.fmuladd.f32(float %51, float 4.000000e+00, float %49)
  %53 = getelementptr inbounds nuw float, ptr %16, i64 %33
  %54 = load float, ptr %53, align 4, !tbaa !10
  %55 = tail call float @llvm.fmuladd.f32(float %54, float 9.000000e+00, float %52)
  %56 = getelementptr inbounds nuw float, ptr %16, i64 %38
  %57 = load float, ptr %56, align 4, !tbaa !10
  %58 = tail call float @llvm.fmuladd.f32(float %57, float 1.200000e+01, float %55)
  %59 = getelementptr inbounds nuw float, ptr %16, i64 %42
  %60 = load float, ptr %59, align 4, !tbaa !10
  %61 = tail call float @llvm.fmuladd.f32(float %60, float 9.000000e+00, float %58)
  %62 = getelementptr inbounds nuw float, ptr %16, i64 %46
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %63, float 4.000000e+00, float %61)
  %65 = getelementptr inbounds nuw float, ptr %17, i64 %30
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = tail call float @llvm.fmuladd.f32(float %66, float 5.000000e+00, float %64)
  %68 = getelementptr inbounds nuw float, ptr %17, i64 %33
  %69 = load float, ptr %68, align 4, !tbaa !10
  %70 = tail call float @llvm.fmuladd.f32(float %69, float 1.200000e+01, float %67)
  %71 = getelementptr inbounds nuw float, ptr %17, i64 %38
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 1.500000e+01, float %70)
  %74 = getelementptr inbounds nuw float, ptr %17, i64 %42
  %75 = load float, ptr %74, align 4, !tbaa !10
  %76 = tail call float @llvm.fmuladd.f32(float %75, float 1.200000e+01, float %73)
  %77 = getelementptr inbounds nuw float, ptr %17, i64 %46
  %78 = load float, ptr %77, align 4, !tbaa !10
  %79 = tail call float @llvm.fmuladd.f32(float %78, float 5.000000e+00, float %76)
  %80 = getelementptr inbounds nuw float, ptr %18, i64 %30
  %81 = load float, ptr %80, align 4, !tbaa !10
  %82 = tail call float @llvm.fmuladd.f32(float %81, float 4.000000e+00, float %79)
  %83 = getelementptr inbounds nuw float, ptr %18, i64 %33
  %84 = load float, ptr %83, align 4, !tbaa !10
  %85 = tail call float @llvm.fmuladd.f32(float %84, float 9.000000e+00, float %82)
  %86 = getelementptr inbounds nuw float, ptr %18, i64 %38
  %87 = load float, ptr %86, align 4, !tbaa !10
  %88 = tail call float @llvm.fmuladd.f32(float %87, float 1.200000e+01, float %85)
  %89 = getelementptr inbounds nuw float, ptr %18, i64 %42
  %90 = load float, ptr %89, align 4, !tbaa !10
  %91 = tail call float @llvm.fmuladd.f32(float %90, float 9.000000e+00, float %88)
  %92 = getelementptr inbounds nuw float, ptr %18, i64 %46
  %93 = load float, ptr %92, align 4, !tbaa !10
  %94 = tail call float @llvm.fmuladd.f32(float %93, float 4.000000e+00, float %91)
  %95 = getelementptr inbounds nuw float, ptr %19, i64 %30
  %96 = load float, ptr %95, align 4, !tbaa !10
  %97 = tail call float @llvm.fmuladd.f32(float %96, float 2.000000e+00, float %94)
  %98 = getelementptr inbounds nuw float, ptr %19, i64 %33
  %99 = load float, ptr %98, align 4, !tbaa !10
  %100 = tail call float @llvm.fmuladd.f32(float %99, float 4.000000e+00, float %97)
  %101 = getelementptr inbounds nuw float, ptr %19, i64 %38
  %102 = load float, ptr %101, align 4, !tbaa !10
  %103 = tail call float @llvm.fmuladd.f32(float %102, float 5.000000e+00, float %100)
  %104 = getelementptr inbounds nuw float, ptr %19, i64 %42
  %105 = load float, ptr %104, align 4, !tbaa !10
  %106 = tail call float @llvm.fmuladd.f32(float %105, float 4.000000e+00, float %103)
  %107 = getelementptr inbounds nuw float, ptr %19, i64 %46
  %108 = load float, ptr %107, align 4, !tbaa !10
  %109 = tail call float @llvm.fmuladd.f32(float %108, float 2.000000e+00, float %106)
  %110 = getelementptr inbounds nuw float, ptr %21, i64 %30
  store float %109, ptr %110, align 4, !tbaa !10
  %111 = add nuw nsw i64 %29, 1
  %112 = icmp eq i64 %111, 64
  br i1 %112, label %25, label %28
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
