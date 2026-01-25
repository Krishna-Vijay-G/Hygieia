'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import Image from 'next/image';
import toast from 'react-hot-toast';
import { 
  User, 
  Mail, 
  Phone, 
  Calendar,
  Shield,
  Save,
  Loader2,
  AlertCircle,
  CheckCircle,
  Upload,
  X
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { authApi } from '@/lib/api';
import { formatDate, getImageURL } from '@/lib/utils';

const profileSchema = z.object({
  first_name: z.string().min(1, 'First name is required'),
  last_name: z.string().min(1, 'Last name is required'),
  email: z.string().email('Invalid email address'),
  phone: z.string().optional(),
  avatar_url: z.string().optional(),
});

type ProfileFormData = z.infer<typeof profileSchema>;

export default function ProfilePage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading, checkAuth } = useAuthStore();
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [isUploadingAvatar, setIsUploadingAvatar] = useState(false);
  const [avatarChanged, setAvatarChanged] = useState(false);
  const [originalAvatar, setOriginalAvatar] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    reset,
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      first_name: user?.first_name || '',
      last_name: user?.last_name || '',
      email: user?.email || '',
      phone: user?.phone || '',
    },
  });

  // Reset form when user data loads
  useEffect(() => {
    if (user) {
      reset({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        email: user.email || '',
        phone: user.phone || '',
      });
      // Only set avatarPreview if it's not currently showing a base64 preview (i.e., not changed by user)
      if (user.avatar_url && !avatarPreview?.startsWith('data:')) {
        setAvatarPreview(user.avatar_url);
      }
      // Set originalAvatar on first load
      if (originalAvatar === null) {
        setOriginalAvatar(user.avatar_url || '');
      }
      // Only reset avatarChanged if the avatar actually changed from what we had
      else if (originalAvatar !== (user.avatar_url || '')) {
        setAvatarChanged(false);
      }
    }
  }, [user, reset, originalAvatar]);

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast.error('Image must be less than 5MB');
      return;
    }

    setIsUploadingAvatar(true);
    const reader = new FileReader();
    
    reader.onload = async (event) => {
      try {
        const base64String = event.target?.result as string;
        setAvatarPreview(base64String);
        setAvatarChanged(true);
        toast.success('Avatar preview updated');
      } catch (err) {
        toast.error('Failed to process image');
      } finally {
        setIsUploadingAvatar(false);
      }
    };

    reader.onerror = () => {
      toast.error('Failed to read file');
      setIsUploadingAvatar(false);
    };

    reader.readAsDataURL(file);
  };

  const removeAvatar = () => {
    setAvatarPreview(null);
    setAvatarChanged(true);
  };

  const mutation = useMutation({
    mutationFn: (data: ProfileFormData) => authApi.updateProfile(data),
    onSuccess: async () => {
      setSuccess(true);
      setError(null);
      // Update originalAvatar to the new saved avatar before refreshing auth
      await checkAuth();
      // After checkAuth, user.avatar_url will be updated with the path
      // Set avatarPreview and originalAvatar to the new path
      if (user?.avatar_url) {
        setAvatarPreview(user.avatar_url);
        setOriginalAvatar(user.avatar_url);
      }
      setAvatarChanged(false); // Reset avatar changed flag
      toast.success('Profile updated successfully!');
      setTimeout(() => setSuccess(false), 3000);
    },
    onError: (error: any) => {
      setError(error.response?.data?.message || 'Failed to update profile');
      setSuccess(false);
    },
  });

  const onSubmit = (data: ProfileFormData) => {
    // If avatar was changed, use the preview instead of the form data
    if (avatarChanged && avatarPreview) {
      data.avatar_url = avatarPreview;
    }
    setError(null);
    setSuccess(false);
    mutation.mutate(data);
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            My Profile
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Manage your account information and settings
          </p>
        </motion.div>

        {/* Profile Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-8 mb-6"
        >
          {/* Avatar Section */}
          <div className="flex flex-col sm:flex-row items-center gap-6 mb-8 pb-8 border-b border-slate-100 dark:border-slate-700">
            <div className="relative">
              <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center overflow-hidden">
                {avatarPreview ? (
                  <img
                    src={avatarPreview.startsWith('/uploads') ? getImageURL(avatarPreview) : avatarPreview}
                    alt="Avatar preview"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-4xl font-bold text-white">
                    {user?.first_name?.[0]}{user?.last_name?.[0]}
                  </span>
                )}
              </div>
              <label
                htmlFor="avatar-input"
                className="absolute -bottom-2 -right-2 p-2 bg-primary-500 hover:bg-primary-600 rounded-full cursor-pointer transition-colors shadow-lg"
              >
                <Upload className="w-4 h-4 text-white" />
                <input
                  id="avatar-input"
                  type="file"
                  accept="image/*"
                  onChange={handleAvatarChange}
                  disabled={isUploadingAvatar}
                  className="hidden"
                />
              </label>
              {avatarPreview && (
                <button
                  type="button"
                  onClick={removeAvatar}
                  className="absolute -top-2 -right-2 p-1 bg-red-500 hover:bg-red-600 rounded-full text-white transition-colors shadow-lg"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <div className="text-center sm:text-left">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
                {user?.first_name} {user?.last_name}
              </h2>
              <p className="text-slate-500 dark:text-slate-400">@{user?.username}</p>
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-3 mt-2">
                {user?.is_admin && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">
                    <Shield className="w-3 h-3" />
                    Admin
                  </span>
                )}
                <span className="text-sm text-slate-500 dark:text-slate-400 flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  Joined {formatDate(user?.created_at || '')}
                </span>
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-3">
                Click the upload icon to change your avatar (optional, max 5MB)
              </p>
            </div>
          </div>

          {/* Alerts */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}
          {success && (
            <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
              <p className="text-green-700 dark:text-green-300">Profile updated successfully!</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  First Name
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    type="text"
                    {...register('first_name')}
                    className="input pl-12 w-full"
                  />
                </div>
                {errors.first_name && (
                  <p className="mt-1 text-sm text-red-600">{errors.first_name.message}</p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Last Name
                </label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    type="text"
                    {...register('last_name')}
                    className="input pl-12 w-full"
                  />
                </div>
                {errors.last_name && (
                  <p className="mt-1 text-sm text-red-600">{errors.last_name.message}</p>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="email"
                  {...register('email')}
                  className="input pl-12 w-full"
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-sm text-red-600">{errors.email.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Phone Number
              </label>
              <div className="relative">
                <Phone className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="tel"
                  {...register('phone')}
                  className="input pl-12 w-full"
                  placeholder="Optional"
                />
              </div>
            </div>

            <div className="flex justify-end pt-4">
              <button
                type="submit"
                disabled={mutation.isPending || (!isDirty && !avatarChanged)}
                className="btn btn-primary px-6"
              >
                {mutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5 mr-2" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          </form>
        </motion.div>

        {/* Account Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6"
        >
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Account Information
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center py-3 border-b border-slate-100 dark:border-slate-700">
              <span className="text-slate-600 dark:text-slate-400">Username</span>
              <span className="font-medium text-slate-900 dark:text-white">@{user?.username}</span>
            </div>
            <div className="flex justify-between items-center py-3 border-b border-slate-100 dark:border-slate-700">
              <span className="text-slate-600 dark:text-slate-400">Account Type</span>
              <span className="font-medium text-slate-900 dark:text-white">
                {user?.is_admin ? 'Administrator' : 'Standard User'}
              </span>
            </div>
            <div className="flex justify-between items-center py-3">
              <span className="text-slate-600 dark:text-slate-400">Account Status</span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                Active
              </span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
