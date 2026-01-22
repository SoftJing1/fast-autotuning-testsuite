; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %5 = getelementptr i8, ptr %0, i64 8224
  %6 = getelementptr i8, ptr %0, i64 12336
  %7 = getelementptr i8, ptr %0, i64 16448
  br label %8

8:                                                ; preds = %3, %18
  %9 = phi i64 [ 0, %3 ], [ %19, %18 ]
  %10 = shl nuw nsw i64 %9, 4
  %11 = add i64 %10, %4
  %12 = add i64 %11, 1
  %13 = add i64 %11, 2
  %14 = add i64 %11, 3
  %15 = add i64 %11, 4
  %16 = getelementptr float, ptr %2, i64 %11
  br label %21

17:                                               ; preds = %18
  ret void

18:                                               ; preds = %21
  %19 = add nuw nsw i64 %9, 1
  %20 = icmp eq i64 %19, 64
  br i1 %20, label %17, label %8

21:                                               ; preds = %8, %21
  %22 = phi i64 [ 0, %8 ], [ %40, %21 ]
  %23 = mul nuw nsw i64 %22, 1028
  %24 = getelementptr float, ptr %0, i64 %23
  %25 = getelementptr float, ptr %24, i64 %11
  %26 = load float, ptr %25, align 4, !tbaa !10
  %27 = getelementptr float, ptr %24, i64 %12
  %28 = load float, ptr %27, align 4, !tbaa !10
  %29 = fmul float %28, 4.000000e+00
  %30 = tail call float @llvm.fmuladd.f32(float %26, float 2.000000e+00, float %29)
  %31 = getelementptr float, ptr %24, i64 %13
  %32 = load float, ptr %31, align 4, !tbaa !10
  %33 = tail call float @llvm.fmuladd.f32(float %32, float 5.000000e+00, float %30)
  %34 = getelementptr float, ptr %24, i64 %14
  %35 = load float, ptr %34, align 4, !tbaa !10
  %36 = tail call float @llvm.fmuladd.f32(float %35, float 4.000000e+00, float %33)
  %37 = getelementptr float, ptr %24, i64 %15
  %38 = load float, ptr %37, align 4, !tbaa !10
  %39 = tail call float @llvm.fmuladd.f32(float %38, float 2.000000e+00, float %36)
  %40 = add nuw nsw i64 %22, 1
  %41 = mul nuw nsw i64 %40, 4112
  %42 = getelementptr i8, ptr %0, i64 %41
  %43 = getelementptr float, ptr %42, i64 %11
  %44 = load float, ptr %43, align 4, !tbaa !10
  %45 = tail call float @llvm.fmuladd.f32(float %44, float 4.000000e+00, float %39)
  %46 = getelementptr float, ptr %42, i64 %12
  %47 = load float, ptr %46, align 4, !tbaa !10
  %48 = tail call float @llvm.fmuladd.f32(float %47, float 9.000000e+00, float %45)
  %49 = getelementptr float, ptr %42, i64 %13
  %50 = load float, ptr %49, align 4, !tbaa !10
  %51 = tail call float @llvm.fmuladd.f32(float %50, float 1.200000e+01, float %48)
  %52 = getelementptr float, ptr %42, i64 %14
  %53 = load float, ptr %52, align 4, !tbaa !10
  %54 = tail call float @llvm.fmuladd.f32(float %53, float 9.000000e+00, float %51)
  %55 = getelementptr float, ptr %42, i64 %15
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = tail call float @llvm.fmuladd.f32(float %56, float 4.000000e+00, float %54)
  %58 = getelementptr float, ptr %5, i64 %23
  %59 = getelementptr float, ptr %58, i64 %11
  %60 = load float, ptr %59, align 4, !tbaa !10
  %61 = tail call float @llvm.fmuladd.f32(float %60, float 5.000000e+00, float %57)
  %62 = getelementptr float, ptr %58, i64 %12
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %63, float 1.200000e+01, float %61)
  %65 = getelementptr float, ptr %58, i64 %13
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = tail call float @llvm.fmuladd.f32(float %66, float 1.500000e+01, float %64)
  %68 = getelementptr float, ptr %58, i64 %14
  %69 = load float, ptr %68, align 4, !tbaa !10
  %70 = tail call float @llvm.fmuladd.f32(float %69, float 1.200000e+01, float %67)
  %71 = getelementptr float, ptr %58, i64 %15
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 5.000000e+00, float %70)
  %74 = getelementptr float, ptr %6, i64 %23
  %75 = getelementptr float, ptr %74, i64 %11
  %76 = load float, ptr %75, align 4, !tbaa !10
  %77 = tail call float @llvm.fmuladd.f32(float %76, float 4.000000e+00, float %73)
  %78 = getelementptr float, ptr %74, i64 %12
  %79 = load float, ptr %78, align 4, !tbaa !10
  %80 = tail call float @llvm.fmuladd.f32(float %79, float 9.000000e+00, float %77)
  %81 = getelementptr float, ptr %74, i64 %13
  %82 = load float, ptr %81, align 4, !tbaa !10
  %83 = tail call float @llvm.fmuladd.f32(float %82, float 1.200000e+01, float %80)
  %84 = getelementptr float, ptr %74, i64 %14
  %85 = load float, ptr %84, align 4, !tbaa !10
  %86 = tail call float @llvm.fmuladd.f32(float %85, float 9.000000e+00, float %83)
  %87 = getelementptr float, ptr %74, i64 %15
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %88, float 4.000000e+00, float %86)
  %90 = getelementptr float, ptr %7, i64 %23
  %91 = getelementptr float, ptr %90, i64 %11
  %92 = load float, ptr %91, align 4, !tbaa !10
  %93 = tail call float @llvm.fmuladd.f32(float %92, float 2.000000e+00, float %89)
  %94 = getelementptr float, ptr %90, i64 %12
  %95 = load float, ptr %94, align 4, !tbaa !10
  %96 = tail call float @llvm.fmuladd.f32(float %95, float 4.000000e+00, float %93)
  %97 = getelementptr float, ptr %90, i64 %13
  %98 = load float, ptr %97, align 4, !tbaa !10
  %99 = tail call float @llvm.fmuladd.f32(float %98, float 5.000000e+00, float %96)
  %100 = getelementptr float, ptr %90, i64 %14
  %101 = load float, ptr %100, align 4, !tbaa !10
  %102 = tail call float @llvm.fmuladd.f32(float %101, float 4.000000e+00, float %99)
  %103 = getelementptr float, ptr %90, i64 %15
  %104 = load float, ptr %103, align 4, !tbaa !10
  %105 = tail call float @llvm.fmuladd.f32(float %104, float 2.000000e+00, float %102)
  %106 = shl nuw nsw i64 %22, 12
  %107 = getelementptr i8, ptr %16, i64 %106
  store float %105, ptr %107, align 4, !tbaa !10
  %108 = icmp eq i64 %40, 1024
  br i1 %108, label %18, label %21
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

attributes #0 = { convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nounwind willreturn memory(none) }

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
