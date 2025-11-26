import React, { useState } from 'react';
import {
  Box,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Button,
  Flex,
  Text,
  useToast,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Progress,
  Select,
  Card,
  CardBody,
  useColorModeValue
} from '@chakra-ui/react';
import { FiMoreVertical, FiRefreshCw, FiXCircle, FiTrash2 } from 'react-icons/fi';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getAllTasks, cancelTask, deleteTask } from '../services/api';

const Tasks = () => {
  const [typeFilter, setTypeFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const toast = useToast();
  const queryClient = useQueryClient();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch tasks
  const {
    data: tasksData,
    isLoading,
    refetch
  } = useQuery({
    queryKey: ['tasks'],
    queryFn: getAllTasks,
    refetchInterval: 3000, // Update every 3 seconds
    refetchOnWindowFocus: true,
    staleTime: 2000
  });

  // Cancel task mutation
  const cancelMutation = useMutation({
    mutationFn: cancelTask,
    onSuccess: () => {
      queryClient.invalidateQueries(['tasks']);
      toast({
        title: 'Task cancelled',
        status: 'info',
        duration: 3000,
        isClosable: true
      });
    },
    onError: (error) => {
      toast({
        title: 'Error cancelling task',
        description: error.response?.data?.detail || error.message,
        status: 'error',
        duration: 3000,
        isClosable: true
      });
    }
  });

  // Delete task mutation
  const deleteTaskMutation = useMutation({
    mutationFn: deleteTask,
    onMutate: async (taskId) => {
      // Cancel current queries
      await queryClient.cancelQueries(['tasks']);

      // Save previous state
      const previousTasks = queryClient.getQueryData(['tasks']);

      // Optimistically update UI
      queryClient.setQueryData(['tasks'], (old) => ({
        ...old,
        tasks: old.tasks.filter((task) => task.id !== taskId),
      }));

      return { previousTasks };
    },
    onError: (err, taskId, context) => {
      // Restore previous state on error
      queryClient.setQueryData(['tasks'], context.previousTasks);

      toast({
        title: 'Error deleting task',
        description: err.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries(['tasks']);
    },
  });

  // Get tasks from the API response
  const tasks = tasksData?.tasks || [];

  // Filter tasks based on type and status
  const filteredTasks = tasks.filter(task => {
    return (
      (typeFilter === '' || task.type === typeFilter) &&
      (statusFilter === '' || task.status === statusFilter)
    );
  });

  // Get unique task types and statuses for filters
  const taskTypes = [...new Set(tasks.map(task => task.type))];
  const taskStatuses = [...new Set(tasks.map(task => task.status))];

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Get color for status badge
  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return 'yellow';
      case 'running': return 'blue';
      case 'completed': return 'green';
      case 'failed': return 'red';
      case 'cancelled': return 'gray';
      default: return 'gray';
    }
  };

  // Handle task cancellation
  const handleCancelTask = (taskId) => {
    if (window.confirm('Are you sure you want to cancel this task?')) {
      cancelMutation.mutate(taskId);
    }
  };

  // Handle task deletion
  const handleDeleteTask = (taskId) => {
    if (window.confirm('Are you sure you want to delete this task? This action cannot be undone.')) {
      deleteTaskMutation.mutate(taskId);
    }
  };

  return (
    <Box>
      <Heading size="md" mb={6}>Tasks</Heading>

      {/* Filters */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6} boxShadow="sm">
        <CardBody>
          <Flex direction={{ base: 'column', md: 'row' }} mb={4} gap={4}>
            <Select
              placeholder="All Types"
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              maxW={{ md: '200px' }}
            >
              {taskTypes.map((type) => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </Select>

            <Select
              placeholder="All Statuses"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              maxW={{ md: '200px' }}
            >
              {taskStatuses.map((status) => (
                <option key={status} value={status}>
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </option>
              ))}
            </Select>

            <Button
              leftIcon={<FiRefreshCw />}
              ml="auto"
              onClick={() => refetch()}
              isLoading={isLoading}
            >
              Refresh
            </Button>
          </Flex>

          <Text color="gray.600" fontSize="sm">
            {filteredTasks.length} tasks found
          </Text>
        </CardBody>
      </Card>

      {/* Tasks Table */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} boxShadow="sm" overflow="hidden">
        <Box overflowX="auto">
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th width="10%" textAlign="center">ID</Th>
                <Th width="15%" textAlign="center">Type</Th>
                <Th width="15%" textAlign="center">Status</Th>
                <Th width="20%" textAlign="center">Progress</Th>
                <Th width="20%" textAlign="center">Created</Th>
                <Th width="20%" textAlign="center">Updated</Th>
                <Th width="80px"></Th>
              </Tr>
            </Thead>
            <Tbody>
              {isLoading ? (
                <Tr>
                  <Td colSpan={7} textAlign="center" py={10}>
                    Loading tasks...
                  </Td>
                </Tr>
              ) : filteredTasks.length === 0 ? (
                <Tr>
                  <Td colSpan={7} textAlign="center" py={10}>
                    No tasks found.
                  </Td>
                </Tr>
              ) : filteredTasks.map((task) => (
                <Tr key={task.id} _hover={{ bg: 'gray.50' }}>
                  <Td textAlign="center">{task.id}</Td>
                  <Td textAlign="center">
                    <Badge>
                      {task.type}
                    </Badge>
                  </Td>
                  <Td textAlign="center">
                    <Badge colorScheme={getStatusColor(task.status)}>
                      {task.status}
                    </Badge>
                  </Td>
                  <Td textAlign="center">
                    <Box>
                      <Progress
                        value={task.progress * 100}
                        size="sm"
                        colorScheme={getStatusColor(task.status)}
                        borderRadius="md"
                        mb={1}
                      />
                      <Text fontSize="xs">
                        {task.progress_details || `${Math.round(task.progress * 100)}%`}
                      </Text>
                    </Box>
                  </Td>
                  <Td textAlign="center">{formatDate(task.created_at)}</Td>
                  <Td textAlign="center">{formatDate(task.updated_at)}</Td>
                  <Td textAlign="center">
                    <Menu>
                      <MenuButton
                        as={IconButton}
                        icon={<FiMoreVertical />}
                        variant="ghost"
                        size="sm"
                        aria-label="Task options"
                      />
                      <MenuList>
                        {task.status === 'running' || task.status === 'pending' ? (
                          <MenuItem
                            icon={<FiXCircle />}
                            onClick={() => handleCancelTask(task.id)}
                          >
                            Cancel
                          </MenuItem>
                        ) : null}
                        <MenuItem
                          icon={<FiTrash2 />}
                          color="red.500"
                          onClick={() => handleDeleteTask(task.id)}
                        >
                          Delete
                        </MenuItem>
                      </MenuList>
                    </Menu>
                  </Td>
                </Tr>
              ))
              }
            </Tbody>
          </Table>
        </Box>
      </Card>
    </Box>
  );
};

export default Tasks; 