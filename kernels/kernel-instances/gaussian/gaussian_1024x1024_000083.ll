; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_group_idj(i32 noundef 0) #3
  %5 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = shl i64 %4, 9
  %7 = add i64 %5, %6
  %8 = getelementptr i8, ptr %0, i64 8224
  %9 = getelementptr i8, ptr %0, i64 12336
  %10 = getelementptr i8, ptr %0, i64 16448
  br label %11

11:                                               ; preds = %3, %21
  %12 = phi i64 [ 0, %3 ], [ %22, %21 ]
  %13 = shl nuw nsw i64 %12, 6
  %14 = add i64 %7, %13
  %15 = add i64 %14, 1
  %16 = add i64 %14, 2
  %17 = add i64 %14, 3
  %18 = add i64 %14, 4
  %19 = getelementptr float, ptr %2, i64 %14
  br label %24

20:                                               ; preds = %21
  ret void

21:                                               ; preds = %24
  %22 = add nuw nsw i64 %12, 1
  %23 = icmp eq i64 %22, 8
  br i1 %23, label %20, label %11

24:                                               ; preds = %11, %24
  %25 = phi i64 [ 0, %11 ], [ %43, %24 ]
  %26 = mul nuw nsw i64 %25, 1028
  %27 = getelementptr float, ptr %0, i64 %26
  %28 = getelementptr float, ptr %27, i64 %14
  %29 = load float, ptr %28, align 4, !tbaa !10
  %30 = getelementptr float, ptr %27, i64 %15
  %31 = load float, ptr %30, align 4, !tbaa !10
  %32 = fmul float %31, 4.000000e+00
  %33 = tail call float @llvm.fmuladd.f32(float %29, float 2.000000e+00, float %32)
  %34 = getelementptr float, ptr %27, i64 %16
  %35 = load float, ptr %34, align 4, !tbaa !10
  %36 = tail call float @llvm.fmuladd.f32(float %35, float 5.000000e+00, float %33)
  %37 = getelementptr float, ptr %27, i64 %17
  %38 = load float, ptr %37, align 4, !tbaa !10
  %39 = tail call float @llvm.fmuladd.f32(float %38, float 4.000000e+00, float %36)
  %40 = getelementptr float, ptr %27, i64 %18
  %41 = load float, ptr %40, align 4, !tbaa !10
  %42 = tail call float @llvm.fmuladd.f32(float %41, float 2.000000e+00, float %39)
  %43 = add nuw nsw i64 %25, 1
  %44 = mul nuw nsw i64 %43, 4112
  %45 = getelementptr i8, ptr %0, i64 %44
  %46 = getelementptr float, ptr %45, i64 %14
  %47 = load float, ptr %46, align 4, !tbaa !10
  %48 = tail call float @llvm.fmuladd.f32(float %47, float 4.000000e+00, float %42)
  %49 = getelementptr float, ptr %45, i64 %15
  %50 = load float, ptr %49, align 4, !tbaa !10
  %51 = tail call float @llvm.fmuladd.f32(float %50, float 9.000000e+00, float %48)
  %52 = getelementptr float, ptr %45, i64 %16
  %53 = load float, ptr %52, align 4, !tbaa !10
  %54 = tail call float @llvm.fmuladd.f32(float %53, float 1.200000e+01, float %51)
  %55 = getelementptr float, ptr %45, i64 %17
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = tail call float @llvm.fmuladd.f32(float %56, float 9.000000e+00, float %54)
  %58 = getelementptr float, ptr %45, i64 %18
  %59 = load float, ptr %58, align 4, !tbaa !10
  %60 = tail call float @llvm.fmuladd.f32(float %59, float 4.000000e+00, float %57)
  %61 = getelementptr float, ptr %8, i64 %26
  %62 = getelementptr float, ptr %61, i64 %14
  %63 = load float, ptr %62, align 4, !tbaa !10
  %64 = tail call float @llvm.fmuladd.f32(float %63, float 5.000000e+00, float %60)
  %65 = getelementptr float, ptr %61, i64 %15
  %66 = load float, ptr %65, align 4, !tbaa !10
  %67 = tail call float @llvm.fmuladd.f32(float %66, float 1.200000e+01, float %64)
  %68 = getelementptr float, ptr %61, i64 %16
  %69 = load float, ptr %68, align 4, !tbaa !10
  %70 = tail call float @llvm.fmuladd.f32(float %69, float 1.500000e+01, float %67)
  %71 = getelementptr float, ptr %61, i64 %17
  %72 = load float, ptr %71, align 4, !tbaa !10
  %73 = tail call float @llvm.fmuladd.f32(float %72, float 1.200000e+01, float %70)
  %74 = getelementptr float, ptr %61, i64 %18
  %75 = load float, ptr %74, align 4, !tbaa !10
  %76 = tail call float @llvm.fmuladd.f32(float %75, float 5.000000e+00, float %73)
  %77 = getelementptr float, ptr %9, i64 %26
  %78 = getelementptr float, ptr %77, i64 %14
  %79 = load float, ptr %78, align 4, !tbaa !10
  %80 = tail call float @llvm.fmuladd.f32(float %79, float 4.000000e+00, float %76)
  %81 = getelementptr float, ptr %77, i64 %15
  %82 = load float, ptr %81, align 4, !tbaa !10
  %83 = tail call float @llvm.fmuladd.f32(float %82, float 9.000000e+00, float %80)
  %84 = getelementptr float, ptr %77, i64 %16
  %85 = load float, ptr %84, align 4, !tbaa !10
  %86 = tail call float @llvm.fmuladd.f32(float %85, float 1.200000e+01, float %83)
  %87 = getelementptr float, ptr %77, i64 %17
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %88, float 9.000000e+00, float %86)
  %90 = getelementptr float, ptr %77, i64 %18
  %91 = load float, ptr %90, align 4, !tbaa !10
  %92 = tail call float @llvm.fmuladd.f32(float %91, float 4.000000e+00, float %89)
  %93 = getelementptr float, ptr %10, i64 %26
  %94 = getelementptr float, ptr %93, i64 %14
  %95 = load float, ptr %94, align 4, !tbaa !10
  %96 = tail call float @llvm.fmuladd.f32(float %95, float 2.000000e+00, float %92)
  %97 = getelementptr float, ptr %93, i64 %15
  %98 = load float, ptr %97, align 4, !tbaa !10
  %99 = tail call float @llvm.fmuladd.f32(float %98, float 4.000000e+00, float %96)
  %100 = getelementptr float, ptr %93, i64 %16
  %101 = load float, ptr %100, align 4, !tbaa !10
  %102 = tail call float @llvm.fmuladd.f32(float %101, float 5.000000e+00, float %99)
  %103 = getelementptr float, ptr %93, i64 %17
  %104 = load float, ptr %103, align 4, !tbaa !10
  %105 = tail call float @llvm.fmuladd.f32(float %104, float 4.000000e+00, float %102)
  %106 = getelementptr float, ptr %93, i64 %18
  %107 = load float, ptr %106, align 4, !tbaa !10
  %108 = tail call float @llvm.fmuladd.f32(float %107, float 2.000000e+00, float %105)
  %109 = shl nuw nsw i64 %25, 12
  %110 = getelementptr i8, ptr %19, i64 %109
  store float %108, ptr %110, align 4, !tbaa !10
  %111 = icmp eq i64 %43, 1024
  br i1 %111, label %21, label %24
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare i64 @_Z12get_group_idj(i32 noundef) local_unnamed_addr #1

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
