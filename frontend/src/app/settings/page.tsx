'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import toast from 'react-hot-toast';
import {
  Settings as SettingsIcon,
  Moon,
  Sun,
  Lock,
  Shield,
  Bell,
  Trash2,
  Save,
  Loader2,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  UserX,
  User,
  Mail
} from 'lucide-react';
import { useAuthStore, useThemeStore } from '@/lib/store';
import { authApi } from '@/lib/api';

const passwordSchema = z.object({
  current_password: z.string().min(1, 'Current password is required'),
  new_password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirm_password: z.string(),
}).refine((data) => data.new_password === data.confirm_password, {
  message: "Passwords don't match",
  path: ['confirm_password'],
});

type PasswordFormData = z.infer<typeof passwordSchema>;

export default function SettingsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading, checkAuth } = useAuthStore();
  const { theme, toggleTheme } = useThemeStore();
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const {
    register: registerPassword,
    handleSubmit: handlePasswordSubmit,
    formState: { errors: passwordErrors },
    reset: resetPassword,
    watch
  } = useForm<PasswordFormData>({
    resolver: zodResolver(passwordSchema),
  });

  const newPassword = watch('new_password', '');

  const passwordRequirements = [
    { text: 'At least 8 characters', met: newPassword.length >= 8 },
    { text: 'One uppercase letter', met: /[A-Z]/.test(newPassword) },
    { text: 'One lowercase letter', met: /[a-z]/.test(newPassword) },
    { text: 'One number', met: /[0-9]/.test(newPassword) },
  ];

  // Password change mutation
  const changePasswordMutation = useMutation({
    mutationFn: (data: PasswordFormData) => authApi.changePassword(data),
    onSuccess: () => {
      toast.success('Password changed successfully');
      resetPassword();
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.error || 'Failed to change password');
    },
  });

  // Deactivate account mutation
  const deactivateMutation = useMutation({
    mutationFn: () => authApi.deactivateAccount(),
    onSuccess: () => {
      toast.success('Account deactivated successfully');
      // Redirect to login after a short delay
      setTimeout(() => {
        router.push('/login');
      }, 2000);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.error || 'Failed to deactivate account');
    },
  });

  const onPasswordSubmit = (data: PasswordFormData) => {
    changePasswordMutation.mutate(data);
  };

  const handleDeactivateAccount = () => {
    deactivateMutation.mutate();
  };

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Settings
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Manage your account preferences and security settings
          </p>
        </motion.div>

        <div className="space-y-8">
          {/* Appearance Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                <SettingsIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              </div>
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                Appearance
              </h2>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800/50 rounded-xl">
                <div className="flex items-center gap-3">
                  {theme === 'dark' ? (
                    <Moon className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                  ) : (
                    <Sun className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                  )}
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">
                      Theme
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Choose your preferred theme
                    </p>
                  </div>
                </div>
                <button
                  onClick={toggleTheme}
                  className="relative inline-flex h-6 w-11 items-center rounded-full bg-slate-200 dark:bg-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      theme === 'dark' ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </motion.div>

          {/* Security Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
                <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
              </div>
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                Security
              </h2>
            </div>

            {/* Change Password */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                  Change Password
                </h3>
                <form onSubmit={handlePasswordSubmit(onPasswordSubmit)} className="space-y-4">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Current Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          {...registerPassword('current_password')}
                          type={showCurrentPassword ? 'text' : 'password'}
                          className="input pl-12 w-full pr-10"
                          placeholder="Enter current password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showCurrentPassword ? (
                            <EyeOff className="h-4 w-4 text-slate-400" />
                          ) : (
                            <Eye className="h-4 w-4 text-slate-400" />
                          )}
                        </button>
                      </div>
                      {passwordErrors.current_password && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {passwordErrors.current_password.message}
                        </p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        New Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          {...registerPassword('new_password')}
                          type={showNewPassword ? 'text' : 'password'}
                          className="input pl-12 w-full pr-10"
                          placeholder="Enter new password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowNewPassword(!showNewPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showNewPassword ? (
                            <EyeOff className="h-4 w-4 text-slate-400" />
                          ) : (
                            <Eye className="h-4 w-4 text-slate-400" />
                          )}
                        </button>
                      </div>
                      {passwordErrors.new_password && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {passwordErrors.new_password.message}
                        </p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Confirm New Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <input
                          {...registerPassword('confirm_password')}
                          type={showConfirmPassword ? 'text' : 'password'}
                          className="input pl-12 w-full pr-10"
                          placeholder="Confirm new password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        >
                          {showConfirmPassword ? (
                            <EyeOff className="h-4 w-4 text-slate-400" />
                          ) : (
                            <Eye className="h-4 w-4 text-slate-400" />
                          )}
                        </button>
                      </div>
                      {passwordErrors.confirm_password && (
                        <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                          {passwordErrors.confirm_password.message}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Password Requirements */}
                  {newPassword && (
                    <div className="p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700">
                      <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                        Password Requirements:
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {passwordRequirements.map((req, index) => (
                          <div key={index} className="flex items-center gap-2">
                            {req.met ? (
                              <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                            ) : (
                              <AlertCircle className="w-4 h-4 text-slate-400 flex-shrink-0" />
                            )}
                            <span className={`text-sm ${req.met ? 'text-green-600 dark:text-green-400' : 'text-slate-500 dark:text-slate-400'}`}>
                              {req.text}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end pt-4">
                    <button
                      type="submit"
                      disabled={changePasswordMutation.isPending}
                      className="btn btn-primary px-6"
                    >
                      {changePasswordMutation.isPending ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin mr-2" />
                          Changing...
                        </>
                      ) : (
                        <>
                          <Save className="w-5 h-5 mr-2" />
                          Change Password
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </motion.div>

          {/* Account Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
                <Shield className="w-5 h-5 text-red-600 dark:text-red-400" />
              </div>
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
                Account
              </h2>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-red-800 dark:text-red-200">
                      Deactivate Account
                    </h3>
                    <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                      Deactivating your account will disable your login access. Your data will be preserved and can be reactivated by an administrator.
                    </p>
                  </div>
                </div>
                <div className="mt-4 flex justify-end">
                  <button
                    onClick={() => setShowDeleteModal(true)}
                    className="btn-secondary px-6 flex items-center gap-2 text-red-700 dark:text-red-300 border-red-300 dark:border-red-600 hover:bg-red-100 dark:hover:bg-red-900/30"
                  >
                    <UserX className="w-4 h-4" />
                    Deactivate Account
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Deactivate Account Modal */}
        {showDeleteModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white dark:bg-slate-800 rounded-2xl p-6 max-w-md w-full"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                  Deactivate Account
                </h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-6">
                Are you sure you want to deactivate your account? This will disable your login access, but your data will be preserved and can be reactivated by an administrator.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowDeleteModal(false)}
                  className="flex-1 btn-secondary"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeactivateAccount}
                  disabled={deactivateMutation.isPending}
                  className="flex-1 btn btn-primary px-6 bg-red-600 hover:bg-red-700 dark:bg-red-600 dark:hover:bg-red-700"
                >
                  {deactivateMutation.isPending ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin mr-2" />
                      Deactivating...
                    </>
                  ) : (
                    <>
                      <UserX className="w-5 h-5 mr-2" />
                      Deactivate
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
}