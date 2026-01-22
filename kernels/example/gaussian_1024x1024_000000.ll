; ModuleID = '/home/jingyu/projects/playground/pyATF-testsuite/fast-autotuning-testsuite/kernel-template/gaussian_static_1.cl'
source_filename = "/home/jingyu/projects/playground/pyATF-testsuite/fast-autotuning-testsuite/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !7 !kernel_arg_access_qual !8 !kernel_arg_type !9 !kernel_arg_base_type !9 !kernel_arg_type_qual !10 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4112
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8224
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 12336
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16448
  br label %8

8:                                                ; preds = %3, %54
  %9 = phi i64 [ 0, %3 ], [ %55, %54 ]
  %10 = mul nuw nsw i64 %9, 1028
  %11 = getelementptr inbounds nuw float, ptr %0, i64 %10
  %12 = getelementptr inbounds nuw float, ptr %4, i64 %10
  %13 = getelementptr inbounds nuw float, ptr %5, i64 %10
  %14 = getelementptr inbounds nuw float, ptr %6, i64 %10
  %15 = getelementptr inbounds nuw float, ptr %7, i64 %10
  %16 = load float, ptr %11, align 4, !tbaa !11
  %17 = getelementptr inbounds nuw i8, ptr %11, i64 4
  %18 = load float, ptr %17, align 4, !tbaa !11
  %19 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %20 = load float, ptr %19, align 4, !tbaa !11
  %21 = getelementptr inbounds nuw i8, ptr %11, i64 12
  %22 = load float, ptr %21, align 4, !tbaa !11
  %23 = load float, ptr %12, align 4, !tbaa !11
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 4
  %25 = load float, ptr %24, align 4, !tbaa !11
  %26 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %27 = load float, ptr %26, align 4, !tbaa !11
  %28 = getelementptr inbounds nuw i8, ptr %12, i64 12
  %29 = load float, ptr %28, align 4, !tbaa !11
  %30 = load float, ptr %13, align 4, !tbaa !11
  %31 = getelementptr inbounds nuw i8, ptr %13, i64 4
  %32 = load float, ptr %31, align 4, !tbaa !11
  %33 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %34 = load float, ptr %33, align 4, !tbaa !11
  %35 = getelementptr inbounds nuw i8, ptr %13, i64 12
  %36 = load float, ptr %35, align 4, !tbaa !11
  %37 = load float, ptr %14, align 4, !tbaa !11
  %38 = getelementptr inbounds nuw i8, ptr %14, i64 4
  %39 = load float, ptr %38, align 4, !tbaa !11
  %40 = getelementptr inbounds nuw i8, ptr %14, i64 8
  %41 = load float, ptr %40, align 4, !tbaa !11
  %42 = getelementptr inbounds nuw i8, ptr %14, i64 12
  %43 = load float, ptr %42, align 4, !tbaa !11
  %44 = load float, ptr %15, align 4, !tbaa !11
  %45 = getelementptr inbounds nuw i8, ptr %15, i64 4
  %46 = load float, ptr %45, align 4, !tbaa !11
  %47 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %48 = load float, ptr %47, align 4, !tbaa !11
  %49 = getelementptr inbounds nuw i8, ptr %15, i64 12
  %50 = load float, ptr %49, align 4, !tbaa !11
  %51 = shl nsw i64 %9, 12
  %52 = getelementptr inbounds nuw i8, ptr %2, i64 %51
  br label %57

53:                                               ; preds = %54
  ret void

54:                                               ; preds = %57
  %55 = add nuw nsw i64 %9, 1
  %56 = icmp eq i64 %55, 1024
  br i1 %56, label %53, label %8

57:                                               ; preds = %8, %57
  %58 = phi float [ %50, %8 ], [ %114, %57 ]
  %59 = phi float [ %48, %8 ], [ %58, %57 ]
  %60 = phi float [ %46, %8 ], [ %59, %57 ]
  %61 = phi float [ %44, %8 ], [ %60, %57 ]
  %62 = phi float [ %43, %8 ], [ %107, %57 ]
  %63 = phi float [ %41, %8 ], [ %62, %57 ]
  %64 = phi float [ %39, %8 ], [ %63, %57 ]
  %65 = phi float [ %37, %8 ], [ %64, %57 ]
  %66 = phi float [ %36, %8 ], [ %100, %57 ]
  %67 = phi float [ %34, %8 ], [ %66, %57 ]
  %68 = phi float [ %32, %8 ], [ %67, %57 ]
  %69 = phi float [ %30, %8 ], [ %68, %57 ]
  %70 = phi float [ %29, %8 ], [ %93, %57 ]
  %71 = phi float [ %27, %8 ], [ %70, %57 ]
  %72 = phi float [ %25, %8 ], [ %71, %57 ]
  %73 = phi float [ %23, %8 ], [ %72, %57 ]
  %74 = phi float [ %22, %8 ], [ %86, %57 ]
  %75 = phi float [ %20, %8 ], [ %74, %57 ]
  %76 = phi float [ %18, %8 ], [ %75, %57 ]
  %77 = phi float [ %16, %8 ], [ %76, %57 ]
  %78 = phi i64 [ 0, %8 ], [ %79, %57 ]
  %79 = add nuw nsw i64 %78, 1
  %80 = fmul float %76, 4.000000e+00
  %81 = tail call float @llvm.fmuladd.f32(float %77, float 2.000000e+00, float %80)
  %82 = tail call float @llvm.fmuladd.f32(float %75, float 5.000000e+00, float %81)
  %83 = tail call float @llvm.fmuladd.f32(float %74, float 4.000000e+00, float %82)
  %84 = add nuw nsw i64 %78, 4
  %85 = getelementptr inbounds nuw float, ptr %11, i64 %84
  %86 = load float, ptr %85, align 4, !tbaa !11
  %87 = tail call float @llvm.fmuladd.f32(float %86, float 2.000000e+00, float %83)
  %88 = tail call float @llvm.fmuladd.f32(float %73, float 4.000000e+00, float %87)
  %89 = tail call float @llvm.fmuladd.f32(float %72, float 9.000000e+00, float %88)
  %90 = tail call float @llvm.fmuladd.f32(float %71, float 1.200000e+01, float %89)
  %91 = tail call float @llvm.fmuladd.f32(float %70, float 9.000000e+00, float %90)
  %92 = getelementptr inbounds nuw float, ptr %12, i64 %84
  %93 = load float, ptr %92, align 4, !tbaa !11
  %94 = tail call float @llvm.fmuladd.f32(float %93, float 4.000000e+00, float %91)
  %95 = tail call float @llvm.fmuladd.f32(float %69, float 5.000000e+00, float %94)
  %96 = tail call float @llvm.fmuladd.f32(float %68, float 1.200000e+01, float %95)
  %97 = tail call float @llvm.fmuladd.f32(float %67, float 1.500000e+01, float %96)
  %98 = tail call float @llvm.fmuladd.f32(float %66, float 1.200000e+01, float %97)
  %99 = getelementptr inbounds nuw float, ptr %13, i64 %84
  %100 = load float, ptr %99, align 4, !tbaa !11
  %101 = tail call float @llvm.fmuladd.f32(float %100, float 5.000000e+00, float %98)
  %102 = tail call float @llvm.fmuladd.f32(float %65, float 4.000000e+00, float %101)
  %103 = tail call float @llvm.fmuladd.f32(float %64, float 9.000000e+00, float %102)
  %104 = tail call float @llvm.fmuladd.f32(float %63, float 1.200000e+01, float %103)
  %105 = tail call float @llvm.fmuladd.f32(float %62, float 9.000000e+00, float %104)
  %106 = getelementptr inbounds nuw float, ptr %14, i64 %84
  %107 = load float, ptr %106, align 4, !tbaa !11
  %108 = tail call float @llvm.fmuladd.f32(float %107, float 4.000000e+00, float %105)
  %109 = tail call float @llvm.fmuladd.f32(float %61, float 2.000000e+00, float %108)
  %110 = tail call float @llvm.fmuladd.f32(float %60, float 4.000000e+00, float %109)
  %111 = tail call float @llvm.fmuladd.f32(float %59, float 5.000000e+00, float %110)
  %112 = tail call float @llvm.fmuladd.f32(float %58, float 4.000000e+00, float %111)
  %113 = getelementptr inbounds nuw float, ptr %15, i64 %84
  %114 = load float, ptr %113, align 4, !tbaa !11
  %115 = tail call float @llvm.fmuladd.f32(float %114, float 2.000000e+00, float %112)
  %116 = getelementptr inbounds nuw float, ptr %52, i64 %78
  store float %115, ptr %116, align 4, !tbaa !11
  %117 = icmp eq i64 %79, 1024
  br i1 %117, label %54, label %57
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!opencl.ocl.version = !{!5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 2, i32 0}
!6 = !{!"clang version 20.1.0"}
!7 = !{i32 1, i32 1, i32 1}
!8 = !{!"none", !"none", !"none"}
!9 = !{!"float*", !"float*", !"float*"}
!10 = !{!"restrict const", !"restrict", !"restrict"}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
