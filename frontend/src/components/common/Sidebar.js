import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Box,
  Flex,
  Text,
  VStack,
  Icon,
  Heading,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  FiHome,
  FiDatabase,
  FiSettings,
  FiCheckCircle,
  FiGrid,
  FiActivity,
} from 'react-icons/fi';

const NavItem = ({ icon, children, to, ...rest }) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  const activeColor = useColorModeValue('blue.500', 'blue.300');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');
  const activeStyle = isActive ? { color: activeColor, fontWeight: 'bold' } : {};
  
  return (
    <Link to={to} style={{ textDecoration: 'none', width: '100%' }}>
      <Flex
        align="center"
        p="4"
        role="group"
        cursor="pointer"
        fontWeight="medium"
        color={useColorModeValue('gray.600', 'gray.200')}
        _hover={{ bg: hoverBg, color: activeColor }}
        borderRadius="md"
        {...activeStyle}
        {...rest}
      >
        {icon && (
          <Icon
            mr="4"
            fontSize="16"
            as={icon}
            color={isActive ? activeColor : 'inherit'}
          />
        )}
        {children}
      </Flex>
    </Link>
  );
};

const Sidebar = () => {
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const textColor = useColorModeValue('gray.600', 'gray.200');
  
  return (
    <Box
      position="fixed"
      left={0}
      width="250px"
      height="100vh"
      bg={bgColor}
      borderRight="1px"
      borderRightColor={borderColor}
      display="flex"
      flexDirection="column"
    >
      <Flex
        h="20"
        alignItems="center"
        mx="8"
        justifyContent="space-between"
      >
        <Heading size="lg" fontWeight="bold">
          Dataset Creator
        </Heading>
      </Flex>
      
      <VStack spacing={0} align="stretch" flex="1">
        <NavItem icon={FiHome} to="/">
          Dashboard
        </NavItem>
        <NavItem icon={FiGrid} to="/generator">
          Generator
        </NavItem>
        <NavItem icon={FiCheckCircle} to="/quality">
          Quality Control
        </NavItem>
        <NavItem icon={FiDatabase} to="/datasets">
          Datasets
        </NavItem>
        <NavItem icon={FiActivity} to="/tasks">
          Tasks
        </NavItem>
        <NavItem icon={FiSettings} to="/settings">
          Settings
        </NavItem>
      </VStack>

      <Box p={4} mt="auto">
        <Text fontSize="xs" color={textColor} lineHeight="1.4">
          Synthetic data for fine-tuning
        </Text>
        <Link
          href="https://github.com/KazKozDev/dataset-creator"
          isExternal
          fontSize="3xs"
          color={textColor}
          _hover={{ color: 'blue.500' }}
          mt={1}
          lineHeight="1.4"
        >
          github.com/kazkozdev
        </Link>
      </Box>
    </Box>
  );
};

export default Sidebar;