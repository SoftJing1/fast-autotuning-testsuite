; ModuleID = 'kernels/kernel-template/gaussian_static_1.cl'
source_filename = "kernels/kernel-template/gaussian_static_1.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite) uwtable
define spir_kernel void @gaussian_1(ptr noalias nocapture noundef readonly align 4 %0, ptr noalias nocapture noundef readnone align 4 %1, ptr noalias nocapture noundef writeonly align 4 %2) local_unnamed_addr #0 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !9 {
  %4 = tail call i64 @_Z12get_group_idj(i32 noundef 0) #3
  %5 = tail call i64 @_Z12get_local_idj(i32 noundef 0) #3
  %6 = shl i64 %4, 9
  %7 = shl i64 %5, 3
  %8 = add i64 %7, %6
  %9 = getelementptr i8, ptr %0, i64 4112
  %10 = getelementptr i8, ptr %0, i64 8224
  %11 = getelementptr i8, ptr %0, i64 12336
  %12 = getelementptr i8, ptr %0, i64 16448
  br label %13

13:                                               ; preds = %3, %28
  %14 = phi i1 [ true, %3 ], [ false, %28 ]
  %15 = phi i64 [ 0, %3 ], [ 256, %28 ]
  %16 = add i64 %8, %15
  br label %18

17:                                               ; preds = %28
  ret void

18:                                               ; preds = %13, %29
  %19 = phi i64 [ 0, %13 ], [ %30, %29 ]
  %20 = mul nuw nsw i64 %19, 1028
  %21 = getelementptr float, ptr %0, i64 %20
  %22 = getelementptr float, ptr %9, i64 %20
  %23 = getelementptr float, ptr %10, i64 %20
  %24 = getelementptr float, ptr %11, i64 %20
  %25 = getelementptr float, ptr %12, i64 %20
  %26 = shl nuw nsw i64 %19, 12
  %27 = getelementptr i8, ptr %2, i64 %26
  br label %32

28:                                               ; preds = %29
  br i1 %14, label %13, label %17

29:                                               ; preds = %32
  %30 = add nuw nsw i64 %19, 1
  %31 = icmp eq i64 %30, 1024
  br i1 %31, label %28, label %18

32:                                               ; preds = %18, %32
  %33 = phi i64 [ 0, %18 ], [ %115, %32 ]
  %34 = add nuw nsw i64 %16, %33
  %35 = getelementptr float, ptr %21, i64 %34
  %36 = load float, ptr %35, align 4, !tbaa !10
  %37 = add i64 %34, 1
  %38 = getelementptr float, ptr %21, i64 %37
  %39 = load float, ptr %38, align 4, !tbaa !10
  %40 = fmul float %39, 4.000000e+00
  %41 = tail call float @llvm.fmuladd.f32(float %36, float 2.000000e+00, float %40)
  %42 = add i64 %34, 2
  %43 = getelementptr float, ptr %21, i64 %42
  %44 = load float, ptr %43, align 4, !tbaa !10
  %45 = tail call float @llvm.fmuladd.f32(float %44, float 5.000000e+00, float %41)
  %46 = add i64 %34, 3
  %47 = getelementptr float, ptr %21, i64 %46
  %48 = load float, ptr %47, align 4, !tbaa !10
  %49 = tail call float @llvm.fmuladd.f32(float %48, float 4.000000e+00, float %45)
  %50 = add i64 %34, 4
  %51 = getelementptr float, ptr %21, i64 %50
  %52 = load float, ptr %51, align 4, !tbaa !10
  %53 = tail call float @llvm.fmuladd.f32(float %52, float 2.000000e+00, float %49)
  %54 = getelementptr float, ptr %22, i64 %34
  %55 = load float, ptr %54, align 4, !tbaa !10
  %56 = tail call float @llvm.fmuladd.f32(float %55, float 4.000000e+00, float %53)
  %57 = getelementptr float, ptr %22, i64 %37
  %58 = load float, ptr %57, align 4, !tbaa !10
  %59 = tail call float @llvm.fmuladd.f32(float %58, float 9.000000e+00, float %56)
  %60 = getelementptr float, ptr %22, i64 %42
  %61 = load float, ptr %60, align 4, !tbaa !10
  %62 = tail call float @llvm.fmuladd.f32(float %61, float 1.200000e+01, float %59)
  %63 = getelementptr float, ptr %22, i64 %46
  %64 = load float, ptr %63, align 4, !tbaa !10
  %65 = tail call float @llvm.fmuladd.f32(float %64, float 9.000000e+00, float %62)
  %66 = getelementptr float, ptr %22, i64 %50
  %67 = load float, ptr %66, align 4, !tbaa !10
  %68 = tail call float @llvm.fmuladd.f32(float %67, float 4.000000e+00, float %65)
  %69 = getelementptr float, ptr %23, i64 %34
  %70 = load float, ptr %69, align 4, !tbaa !10
  %71 = tail call float @llvm.fmuladd.f32(float %70, float 5.000000e+00, float %68)
  %72 = getelementptr float, ptr %23, i64 %37
  %73 = load float, ptr %72, align 4, !tbaa !10
  %74 = tail call float @llvm.fmuladd.f32(float %73, float 1.200000e+01, float %71)
  %75 = getelementptr float, ptr %23, i64 %42
  %76 = load float, ptr %75, align 4, !tbaa !10
  %77 = tail call float @llvm.fmuladd.f32(float %76, float 1.500000e+01, float %74)
  %78 = getelementptr float, ptr %23, i64 %46
  %79 = load float, ptr %78, align 4, !tbaa !10
  %80 = tail call float @llvm.fmuladd.f32(float %79, float 1.200000e+01, float %77)
  %81 = getelementptr float, ptr %23, i64 %50
  %82 = load float, ptr %81, align 4, !tbaa !10
  %83 = tail call float @llvm.fmuladd.f32(float %82, float 5.000000e+00, float %80)
  %84 = getelementptr float, ptr %24, i64 %34
  %85 = load float, ptr %84, align 4, !tbaa !10
  %86 = tail call float @llvm.fmuladd.f32(float %85, float 4.000000e+00, float %83)
  %87 = getelementptr float, ptr %24, i64 %37
  %88 = load float, ptr %87, align 4, !tbaa !10
  %89 = tail call float @llvm.fmuladd.f32(float %88, float 9.000000e+00, float %86)
  %90 = getelementptr float, ptr %24, i64 %42
  %91 = load float, ptr %90, align 4, !tbaa !10
  %92 = tail call float @llvm.fmuladd.f32(float %91, float 1.200000e+01, float %89)
  %93 = getelementptr float, ptr %24, i64 %46
  %94 = load float, ptr %93, align 4, !tbaa !10
  %95 = tail call float @llvm.fmuladd.f32(float %94, float 9.000000e+00, float %92)
  %96 = getelementptr float, ptr %24, i64 %50
  %97 = load float, ptr %96, align 4, !tbaa !10
  %98 = tail call float @llvm.fmuladd.f32(float %97, float 4.000000e+00, float %95)
  %99 = getelementptr float, ptr %25, i64 %34
  %100 = load float, ptr %99, align 4, !tbaa !10
  %101 = tail call float @llvm.fmuladd.f32(float %100, float 2.000000e+00, float %98)
  %102 = getelementptr float, ptr %25, i64 %37
  %103 = load float, ptr %102, align 4, !tbaa !10
  %104 = tail call float @llvm.fmuladd.f32(float %103, float 4.000000e+00, float %101)
  %105 = getelementptr float, ptr %25, i64 %42
  %106 = load float, ptr %105, align 4, !tbaa !10
  %107 = tail call float @llvm.fmuladd.f32(float %106, float 5.000000e+00, float %104)
  %108 = getelementptr float, ptr %25, i64 %46
  %109 = load float, ptr %108, align 4, !tbaa !10
  %110 = tail call float @llvm.fmuladd.f32(float %109, float 4.000000e+00, float %107)
  %111 = getelementptr float, ptr %25, i64 %50
  %112 = load float, ptr %111, align 4, !tbaa !10
  %113 = tail call float @llvm.fmuladd.f32(float %112, float 2.000000e+00, float %110)
  %114 = getelementptr float, ptr %27, i64 %34
  store float %113, ptr %114, align 4, !tbaa !10
  %115 = add nuw nsw i64 %33, 1
  %116 = icmp eq i64 %115, 8
  br i1 %116, label %29, label %32
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
