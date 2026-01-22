; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_group_idj(i32 noundef 0) #3
  %5 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = shl i64 %4, 9
  %7 = shl i64 %5, 5
  %8 = add i64 %7, %6
  %9 = getelementptr i8, ptr %0, i64 4112
  %10 = getelementptr i8, ptr %0, i64 8224
  %11 = getelementptr i8, ptr %0, i64 12336
  %12 = getelementptr i8, ptr %0, i64 16448
  br label %13

13:                                               ; preds = %3, %24
  %14 = phi i64 [ 0, %3 ], [ %25, %24 ]
  %15 = mul nuw nsw i64 %14, 1028
  %16 = getelementptr float, ptr %0, i64 %15
  %17 = getelementptr float, ptr %9, i64 %15
  %18 = getelementptr float, ptr %10, i64 %15
  %19 = getelementptr float, ptr %11, i64 %15
  %20 = getelementptr float, ptr %12, i64 %15
  %21 = shl nuw nsw i64 %14, 12
  %22 = getelementptr i8, ptr %2, i64 %21
  br label %27

23:                                               ; preds = %24
  ret void

24:                                               ; preds = %27
  %25 = add nuw nsw i64 %14, 1
  %26 = icmp eq i64 %25, 1024
  br i1 %26, label %23, label %13

27:                                               ; preds = %13, %27
  %28 = phi i64 [ 0, %13 ], [ %110, %27 ]
  %29 = add nuw nsw i64 %28, %8
  %30 = getelementptr float, ptr %16, i64 %29
  %31 = load float, ptr %30, align 4, !tbaa !10
  %32 = add i64 %29, 1
  %33 = getelementptr float, ptr %16, i64 %32
  %34 = load float, ptr %33, align 4, !tbaa !10
  %35 = fmul float %34, 4.000000e+00
  %36 = tail call float @llvm.fmuladd.f32(float %31, float 2.000000e+00, float %35)
  %37 = add i64 %29, 2
  %38 = getelementptr float, ptr %16, i64 %37
  %39 = load float, ptr %38, align 4, !tbaa !10
  %40 = tail call float @llvm.fmuladd.f32(float %39, float 5.000000e+00, float %36)
  %41 = add i64 %29, 3
  %42 = getelementptr float, ptr %16, i64 %41
  %43 = load float, ptr %42, align 4, !tbaa !10
  %44 = tail call float @llvm.fmuladd.f32(float %43, float 4.000000e+00, float %40)
  %45 = add i64 %29, 4
  %46 = getelementptr float, ptr %16, i64 %45
  %47 = load float, ptr %46, align 4, !tbaa !10
  %48 = tail call float @llvm.fmuladd.f32(float %47, float 2.000000e+00, float %44)
  %49 = getelementptr float, ptr %17, i64 %29
  %50 = load float, ptr %49, align 4, !tbaa !10
  %51 = tail call float @llvm.fmuladd.f32(float %50, float 4.000000e+00, float %48)
  %52 = getelementptr float, ptr %17, i64 %32
  %53 = load float, ptr %52, align 4, !tbaa !10
  %54 = tail call float @llvm.fmuladd.f32(float %53, float 9.000000e+00, float %51)
  %55 = getelementptr float, ptr %17, i64 %37
  %56 = load float, ptr %55, align 4, !tbaa !10
  %57 = tail call float @llvm.fmuladd.f32(float %56, float 1.200000e+01, float %54)
  %58 = getelementptr float, ptr %17, i64 %41
  %59 = load float, ptr %58, align 4, !tbaa !10
  %60 = tail call float @llvm.fmuladd.f32(float %59, float 9.000000e+00, float %57)
  %61 = getelementptr float, ptr %17, i64 %45
  %62 = load float, ptr %61, align 4, !tbaa !10
  %63 = tail call float @llvm.fmuladd.f32(float %62, float 4.000000e+00, float %60)
  %64 = getelementptr float, ptr %18, i64 %29
  %65 = load float, ptr %64, align 4, !tbaa !10
  %66 = tail call float @llvm.fmuladd.f32(float %65, float 5.000000e+00, float %63)
  %67 = getelementptr float, ptr %18, i64 %32
  %68 = load float, ptr %67, align 4, !tbaa !10
  %69 = tail call float @llvm.fmuladd.f32(float %68, float 1.200000e+01, float %66)
  %70 = getelementptr float, ptr %18, i64 %37
  %71 = load float, ptr %70, align 4, !tbaa !10
  %72 = tail call float @llvm.fmuladd.f32(float %71, float 1.500000e+01, float %69)
  %73 = getelementptr float, ptr %18, i64 %41
  %74 = load float, ptr %73, align 4, !tbaa !10
  %75 = tail call float @llvm.fmuladd.f32(float %74, float 1.200000e+01, float %72)
  %76 = getelementptr float, ptr %18, i64 %45
  %77 = load float, ptr %76, align 4, !tbaa !10
  %78 = tail call float @llvm.fmuladd.f32(float %77, float 5.000000e+00, float %75)
  %79 = getelementptr float, ptr %19, i64 %29
  %80 = load float, ptr %79, align 4, !tbaa !10
  %81 = tail call float @llvm.fmuladd.f32(float %80, float 4.000000e+00, float %78)
  %82 = getelementptr float, ptr %19, i64 %32
  %83 = load float, ptr %82, align 4, !tbaa !10
  %84 = tail call float @llvm.fmuladd.f32(float %83, float 9.000000e+00, float %81)
  %85 = getelementptr float, ptr %19, i64 %37
  %86 = load float, ptr %85, align 4, !tbaa !10
  %87 = tail call float @llvm.fmuladd.f32(float %86, float 1.200000e+01, float %84)
  %88 = getelementptr float, ptr %19, i64 %41
  %89 = load float, ptr %88, align 4, !tbaa !10
  %90 = tail call float @llvm.fmuladd.f32(float %89, float 9.000000e+00, float %87)
  %91 = getelementptr float, ptr %19, i64 %45
  %92 = load float, ptr %91, align 4, !tbaa !10
  %93 = tail call float @llvm.fmuladd.f32(float %92, float 4.000000e+00, float %90)
  %94 = getelementptr float, ptr %20, i64 %29
  %95 = load float, ptr %94, align 4, !tbaa !10
  %96 = tail call float @llvm.fmuladd.f32(float %95, float 2.000000e+00, float %93)
  %97 = getelementptr float, ptr %20, i64 %32
  %98 = load float, ptr %97, align 4, !tbaa !10
  %99 = tail call float @llvm.fmuladd.f32(float %98, float 4.000000e+00, float %96)
  %100 = getelementptr float, ptr %20, i64 %37
  %101 = load float, ptr %100, align 4, !tbaa !10
  %102 = tail call float @llvm.fmuladd.f32(float %101, float 5.000000e+00, float %99)
  %103 = getelementptr float, ptr %20, i64 %41
  %104 = load float, ptr %103, align 4, !tbaa !10
  %105 = tail call float @llvm.fmuladd.f32(float %104, float 4.000000e+00, float %102)
  %106 = getelementptr float, ptr %20, i64 %45
  %107 = load float, ptr %106, align 4, !tbaa !10
  %108 = tail call float @llvm.fmuladd.f32(float %107, float 2.000000e+00, float %105)
  %109 = getelementptr float, ptr %22, i64 %29
  store float %108, ptr %109, align 4, !tbaa !10
  %110 = add nuw nsw i64 %28, 1
  %111 = icmp eq i64 %110, 32
  br i1 %111, label %24, label %27
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
